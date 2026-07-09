#!/usr/bin/env python3
"""BidKV 5 策略对比实验 — 所有策略使用完全相同的 1000 请求 (seed=42, inf rate)."""
import asyncio, json, os, signal, statistics, subprocess, time
import aiohttp

MODEL = "/data/shared_models/Qwen2.5-7B-Instruct"
DATASET = "/data/shared_datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
PROMPTS_FILE = "/tmp/fixed_prompts.json"
SEED, N = 42, 1000
PORT, NPU = 8000, 3

STRATEGIES = ["pe", "pe-sjf", "static-random", "largest-first", "bidkv"]

ENV = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": str(NPU),
    "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/nnal/atb/8.5.1/atb/cxx_abi_1/lib:" + os.environ.get("LD_LIBRARY_PATH","")}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
LOG_DIR = os.path.join(THIS_DIR, "logs")

def prepare_prompts():
    if os.path.exists(PROMPTS_FILE): return
    import random; random.seed(SEED)
    with open(DATASET) as f: data = json.load(f)
    convs = [d for d in data if d.get("conversations")]; random.shuffle(convs)
    prompts = []
    for c in convs:
        for t in c["conversations"]:
            if t.get("from") == "human" and 10 < len(t["value"]) < 32000:
                prompts.append(t["value"]); break
        if len(prompts) >= N: break
    with open(PROMPTS_FILE, "w") as f: json.dump(prompts, f)
    print(f"[prepare] {len(prompts)} prompts saved")

def kill_server():
    for pid in subprocess.check_output(["ps","-eo","pid,cmd"],text=True).splitlines():
        if "vllm" in pid and ("serve" in pid or "EngineCore" in pid):
            try: os.kill(int(pid.split()[0]), signal.SIGKILL)
            except: pass
    time.sleep(5)

def start_server(strategy: str):
    kill_server()
    cfg = json.dumps({
        "recompute_scheduler_enable": True,
        "enable_utility_victim_selection": True,
        "utility_strategy": strategy,
        "utility_kv_gate": 0.0,
    })
    log = os.path.join(LOG_DIR, f"vllm_{strategy}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    proc = subprocess.Popen([
        "vllm","serve",MODEL,"--host","0.0.0.0","--port",str(PORT),
        "--trust-remote-code","--gpu-memory-utilization","0.255",
        "--max-model-len","8192","--max-num-seqs","32",
        "--no-enable-prefix-caching","--additional-config",cfg,
    ], env=ENV, stdout=open(log,"w"), stderr=subprocess.STDOUT)
    print(f"[server] {strategy} PID={proc.pid} log={log}")
    for _ in range(180):
        try:
            with open(log) as f:
                if "Application startup complete" in f.read():
                    time.sleep(3); print(f"[server] {strategy} ready"); return proc, log
        except: pass
        time.sleep(1)
    raise RuntimeError(f"{strategy} failed to start")

async def bench(strategy: str):
    with open(PROMPTS_FILE) as f: prompts = json.load(f)
    results = []; t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0,force_close=True)) as sess:
        async def send(p):
            t_req = time.perf_counter()
            try:
                async with sess.post(f"http://localhost:{PORT}/v1/completions",
                    json={"model":MODEL,"prompt":p,"max_tokens":128,"temperature":0,"stream":False},
                    timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    data = await resp.json()
                    elapsed = (time.perf_counter() - t_req) * 1000
                    tok = data.get("usage",{}).get("completion_tokens",0)
                    results.append({"total_ms":elapsed,"tokens":tok,"error":""})
            except Exception as e:
                results.append({"total_ms":0,"tokens":0,"error":str(e)[:200]})
        tasks = [asyncio.create_task(send(p)) for p in prompts]
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    ok = [r for r in results if not r["error"]]
    total_ms = sorted(r["total_ms"] for r in ok); tok = sum(r["tokens"] for r in ok)
    def pct(p): return total_ms[int(len(total_ms)*p/100)] if total_ms else 0
    summary = {
        "strategy": strategy, "seed": SEED, "num_prompts": N,
        "completed": len(ok), "failed": len(results)-len(ok),
        "total_tokens": tok, "duration_s": elapsed,
        "throughput_tok_s": tok/elapsed if elapsed else 0,
        "e2e_p25": pct(25), "e2e_p50": pct(50), "e2e_p95": pct(95), "e2e_p99": pct(99),
    }
    fname = os.path.join(RESULTS_DIR, f"{strategy}_seed42.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(fname, "w") as f: json.dump({"summary":summary,"results":results}, f)
    print(f"[bench] {strategy}: {len(ok)}/{N} ok, tp={summary['throughput_tok_s']:.0f}tok/s, "
          f"p50={summary['e2e_p50']:.0f}ms p95={summary['e2e_p95']:.0f}ms p99={summary['e2e_p99']:.0f}ms")
    return summary

async def main():
    prepare_prompts()
    all_results = []
    for s in STRATEGIES:
        proc, log = start_server(s)
        try: summary = await bench(s); all_results.append(summary)
        finally: kill_server()

    print("\n" + "="*70)
    print(f"{'Strategy':<18} {'Throughput':>10} {'e2e P50':>10} {'e2e P95':>10} {'e2e P99':>10}")
    print("-"*70)
    for s in all_results:
        print(f"{s['strategy']:<18} {s['throughput_tok_s']:>10.0f} {s['e2e_p50']:>10.0f} {s['e2e_p95']:>10.0f} {s['e2e_p99']:>10.0f}")
    print("="*70)

    # 保存汇总
    with open(os.path.join(RESULTS_DIR, "comparison_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())

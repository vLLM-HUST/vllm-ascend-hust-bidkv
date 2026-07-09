#!/usr/bin/env python3
"""BidKV 5 策略高压对比 — KV cache ~8K tokens (gpu_mem=0.252), inf rate."""
import asyncio, json, os, signal, subprocess, time
import aiohttp

MODEL = "/data/shared_models/Qwen2.5-7B-Instruct"
PROMPTS_FILE = "/tmp/fixed_prompts.json"
SEED, N = 42, 1000
PORT, NPU = 8000, 3
GPU_MEM = "0.252"

STRATEGIES = ["largest-first", "bidkv"]

ENV = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": str(NPU),
    "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/nnal/atb/8.5.1/atb/cxx_abi_1/lib:" + os.environ.get("LD_LIBRARY_PATH","")}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results_hp")
LOG_DIR = os.path.join(THIS_DIR, "logs_hp")
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

def kill_server():
    for pid in subprocess.check_output(["ps","-eo","pid,cmd"],text=True).splitlines():
        if "vllm" in pid and ("serve" in pid or "EngineCore" in pid):
            try: os.kill(int(pid.split()[0]), signal.SIGKILL)
            except: pass
    # 等 NPU HBM 释放（最多 60 秒）
    for _ in range(60):
        time.sleep(1)
        try:
            out = subprocess.check_output(["npu-smi","info","-t","usages","-i",str(NPU)],
                env={"LD_LIBRARY_PATH":ENV["LD_LIBRARY_PATH"]}, text=True, timeout=5)
            for line in out.splitlines():
                if "HBM Usage" in line:
                    pct = int(line.split(":")[-1].strip().rstrip("%"))
                    if pct < 15:
                        print(f"[kill] HBM {pct}%, ready"); return
        except: pass
    print("[kill] timeout waiting for HBM, proceeding anyway")

def start_server(strategy: str):
    kill_server()
    cfg = json.dumps({"recompute_scheduler_enable":True,"enable_utility_victim_selection":True,"utility_strategy":strategy,"utility_kv_gate":0.0})
    log = os.path.join(LOG_DIR, f"vllm_{strategy}.log")
    for attempt in range(3):
        if attempt > 0:
            print(f"[server] {strategy} retry {attempt+1}/3")
            kill_server()
        proc = subprocess.Popen([
            "vllm","serve",MODEL,"--host","0.0.0.0","--port",str(PORT),
            "--trust-remote-code","--gpu-memory-utilization",GPU_MEM,
            "--max-model-len","8192","--max-num-seqs","32",
            "--no-enable-prefix-caching","--additional-config",cfg,
        ], env=ENV, stdout=open(log,"w"), stderr=subprocess.STDOUT)
        print(f"[server] {strategy} PID={proc.pid}")
        for _ in range(180):
            try:
                with open(log) as f:
                    if "Application startup complete" in f.read():
                        time.sleep(3); print(f"[server] {strategy} ready"); return proc, log
            except: pass
            time.sleep(1)
    raise RuntimeError(f"{strategy} failed after 3 attempts")

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
        "e2e_p50": pct(50), "e2e_p95": pct(95), "e2e_p99": pct(99),
    }
    fname = os.path.join(RESULTS_DIR, f"{strategy}_seed42.json")
    with open(fname, "w") as f: json.dump({"summary":summary,"results":results}, f)
    pevts = 0
    try:
        with open(os.path.join(LOG_DIR,f"vllm_{strategy}.log")) as lf:
            pevts = sum(1 for l in lf if "UTILITY_ACTIVE" in l or "FALLBACK" in l or "RANDOM" in l or "LARGEST_FIRST" in l or "PE" in l)
    except: pass
    print(f"[bench] {strategy}: {len(ok)}/{N} ok, tp={summary['throughput_tok_s']:.0f}tok/s, "
          f"p95={summary['e2e_p95']:.0f}ms p99={summary['e2e_p99']:.0f}ms preempt={pevts}")
    return summary

async def main():
    all_results = []
    for s in STRATEGIES:
        proc, log = start_server(s)
        try: summary = await bench(s); all_results.append(summary)
        finally: kill_server()

    print("\n" + "="*72)
    print(f"  GPU Mem={GPU_MEM} | KV ~8.2K tokens | inf rate | seed={SEED}")
    print("="*72)
    print(f"{'Strategy':<18} {'Thru(tok/s)':>10} {'e2e P50':>10} {'e2e P95':>10} {'e2e P99':>10}")
    print("-"*72)
    best_tp = max(s['throughput_tok_s'] for s in all_results)
    best_p95 = min(s['e2e_p95'] for s in all_results)
    best_p99 = min(s['e2e_p99'] for s in all_results)
    for s in all_results:
        tp = f"**{int(s['throughput_tok_s'])}**" if s['throughput_tok_s']==best_tp else str(int(s['throughput_tok_s']))
        p95 = f"**{int(s['e2e_p95'])}**" if s['e2e_p95']==best_p95 else str(int(s['e2e_p95']))
        p99 = f"**{int(s['e2e_p99'])}**" if s['e2e_p99']==best_p99 else str(int(s['e2e_p99']))
        print(f"{s['strategy']:<18} {tp:>10} {int(s['e2e_p50']):>10} {p95:>10} {p99:>10}")
    print("="*72)
    with open(os.path.join(RESULTS_DIR, "comparison_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    asyncio.run(main())

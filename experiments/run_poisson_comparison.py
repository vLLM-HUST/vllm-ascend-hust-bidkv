#!/usr/bin/env python3
"""BidKV 5 策略 Poisson 对比 — block_size=16, Poisson 5.7, seed=42."""
import asyncio, json, os, random, signal, subprocess, time
import aiohttp, numpy as np

MODEL = "/data/shared_models/Qwen2.5-7B-Instruct"
PROMPTS_FILE = "/tmp/fixed_prompts.json"
SEED, N, RATE = 42, 1000, 12.0
PORT, NPU = 8000, 2
GPU_MEM = "0.255"
BLOCK_SIZE = "16"

STRATEGIES = ["pe", "pe-sjf", "static-random", "largest-first", "bidkv"]

ENV = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": str(NPU),
    "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/nnal/atb/8.5.1/atb/cxx_abi_1/lib:" + os.environ.get("LD_LIBRARY_PATH","")}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results_poisson")
LOG_DIR = os.path.join(THIS_DIR, "logs_poisson")
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

def kill_server():
    for pid in subprocess.check_output(["ps","-eo","pid,cmd"],text=True).splitlines():
        if "vllm" in pid and ("serve" in pid or "EngineCore" in pid):
            try: os.kill(int(pid.split()[0]), signal.SIGKILL)
            except: pass
    for _ in range(60):
        time.sleep(1)
        try:
            out = subprocess.check_output(["npu-smi","info","-t","usages","-i",str(NPU)],
                env={"LD_LIBRARY_PATH":ENV["LD_LIBRARY_PATH"]}, text=True, timeout=5)
            for line in out.splitlines():
                if "HBM Usage" in line:
                    if int(line.split(":")[-1].strip().rstrip("%")) < 15: return
        except: pass

def start_server(strategy: str):
    kill_server()
    cfg = json.dumps({"recompute_scheduler_enable":True,"enable_utility_victim_selection":True,"utility_strategy":strategy,"utility_kv_gate":0.95})
    log = os.path.join(LOG_DIR, f"vllm_{strategy}.log")
    for attempt in range(3):
        if attempt > 0: print(f"retry {attempt+1}"); kill_server()
        proc = subprocess.Popen([
            "vllm","serve",MODEL,"--host","0.0.0.0","--port",str(PORT),
            "--trust-remote-code","--block-size",BLOCK_SIZE,
            "--gpu-memory-utilization",GPU_MEM,"--max-model-len","8192",
            "--max-num-seqs","32","--no-enable-prefix-caching",
            "--additional-config",cfg,
        ], env=ENV, stdout=open(log,"w"), stderr=subprocess.STDOUT)
        print(f"[{strategy}] PID={proc.pid}")
        for _ in range(180):
            try:
                with open(log) as lf:
                    if "Application startup complete" in lf.read():
                        time.sleep(3); return proc, log
            except: pass
            time.sleep(1)
    raise RuntimeError(f"{strategy} failed")

async def bench(strategy: str):
    with open(PROMPTS_FILE) as f: prompts = json.load(f)
    np.random.seed(SEED)
    intervals = np.random.exponential(1.0/RATE, size=len(prompts))
    results = []; t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0,force_close=True)) as sess:
        sem = asyncio.Semaphore(500)
        async def send(p):
            async with sem:
                t_req = time.perf_counter()
                try:
                    async with sess.post(f"http://localhost:{PORT}/v1/completions",
                        json={"model":MODEL,"prompt":p,"max_tokens":256,"temperature":0,"stream":False},
                        timeout=aiohttp.ClientTimeout(total=300)) as resp:
                        data = await resp.json(); elapsed = (time.perf_counter()-t_req)*1000
                        tok = data.get("usage",{}).get("completion_tokens",0)
                        results.append({"total_ms":elapsed,"tokens":tok,"error":""})
                except Exception as e:
                    results.append({"total_ms":0,"tokens":0,"error":str(e)[:200]})
        tasks = []
        for i,(p,iv) in enumerate(zip(prompts, intervals)):
            if i > 0: await asyncio.sleep(iv)
            tasks.append(asyncio.create_task(send(p)))
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter()-t0
    ok = [r for r in results if not r["error"]]
    total_ms = sorted(r["total_ms"] for r in ok); tok = sum(r["tokens"] for r in ok)
    def pct(p): return total_ms[int(len(total_ms)*p/100)] if total_ms else 0
    summary = {"strategy":strategy,"rate":RATE,"seed":SEED,"num_prompts":N,
        "completed":len(ok),"failed":len(results)-len(ok),"total_tokens":tok,
        "duration_s":elapsed,"throughput_tok_s":tok/elapsed if elapsed else 0,
        "e2e_p50":pct(50),"e2e_p95":pct(95),"e2e_p99":pct(99)}
    fname = os.path.join(RESULTS_DIR,f"{strategy}_seed42_rate{RATE}.json")
    with open(fname,"w") as f: json.dump({"summary":summary},f)
    print(f"[{strategy}] {len(ok)}/{N} ok, tp={summary['throughput_tok_s']:.0f}t/s "
          f"p50={summary['e2e_p50']:.0f}ms p95={summary['e2e_p95']:.0f}ms p99={summary['e2e_p99']:.0f}ms")
    return summary

async def main():
    all_r = []
    for s in STRATEGIES:
        proc, log = start_server(s)
        try: summary = await bench(s); all_r.append(summary)
        finally: kill_server()
    print(f"\n{'='*65}")
    print(f"  block_size={BLOCK_SIZE} | Poisson {RATE} | seed={SEED} | KV~11.6K")
    print(f"{'='*65}")
    print(f"{'Strategy':<18} {'Thru':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print("-"*50)
    bt=max(s['throughput_tok_s'] for s in all_r); bp95=min(s['e2e_p95'] for s in all_r); bp99=min(s['e2e_p99'] for s in all_r)
    for s in all_r:
        tp=f"**{int(s['throughput_tok_s'])}**" if s['throughput_tok_s']==bt else str(int(s['throughput_tok_s']))
        p95=f"**{int(s['e2e_p95'])}**" if s['e2e_p95']==bp95 else str(int(s['e2e_p95']))
        p99=f"**{int(s['e2e_p99'])}**" if s['e2e_p99']==bp99 else str(int(s['e2e_p99']))
        print(f"{s['strategy']:<18} {tp:>8} {int(s['e2e_p50']):>8} {p95:>8} {p99:>8}")
    with open(os.path.join(RESULTS_DIR,"summary.json"),"w") as f: json.dump(all_r,f)
    print(f"\n→ {RESULTS_DIR}/")

if __name__ == "__main__":
    asyncio.run(main())

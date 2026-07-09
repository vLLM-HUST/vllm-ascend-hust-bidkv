#!/usr/bin/env python3
"""单策略 benchmark — Poisson 到达，streaming 模式测 TTFT/TPOT/e2e."""
import asyncio, json, os, sys, time
import numpy as np
import aiohttp

PROMPTS_FILE = "/tmp/fixed_prompts.json"
MODEL = "/data/shared_models/Qwen2.5-7B-Instruct"
SEED, N, RATE = 42, 1000, 12.0


async def bench(strategy: str, port: int, out_dir: str, rate: float = RATE, max_tokens: int = 256):
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    np.random.seed(SEED)
    intervals = np.random.exponential(1.0 / rate, size=len(prompts))
    results = []
    t0 = time.perf_counter()

    conn = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(connector=conn) as sess:
        sem = asyncio.Semaphore(500)

        async def send(p):
            async with sem:
                t_req = time.perf_counter()
                try:
                    async with sess.post(
                        f"http://localhost:{port}/v1/completions",
                        json={"model": MODEL, "prompt": p, "max_tokens": max_tokens,
                              "temperature": 0, "stream": True},
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        ttft = None
                        token_count = 0
                        async for line in resp.content:
                            line = line.decode("utf-8").strip()
                            if line.startswith("data: "):
                                ds = line[6:]
                                if ds == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(ds)
                                except json.JSONDecodeError:
                                    continue
                                if ttft is None:
                                    ttft = (time.perf_counter() - t_req) * 1000
                                if "choices" in chunk and chunk["choices"]:
                                    token_count += 1
                        e2e = (time.perf_counter() - t_req) * 1000
                        if ttft is None:
                            ttft = e2e
                        tpot = (e2e - ttft) / max(token_count - 1, 1) if token_count > 1 else 0
                        results.append({
                            "ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e,
                            "tokens": token_count, "error": "",
                        })
                except Exception as e:
                    results.append({
                        "ttft_ms": 0, "tpot_ms": 0, "e2e_ms": 0,
                        "tokens": 0, "error": str(e)[:200],
                    })

        tasks = []
        for i, (p, iv) in enumerate(zip(prompts, intervals)):
            if i > 0:
                await asyncio.sleep(iv)
            tasks.append(asyncio.create_task(send(p)))
        await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t0
    ok = [r for r in results if not r["error"]]
    ttft_list = sorted(r["ttft_ms"] for r in ok)
    tpot_list = sorted(r["tpot_ms"] for r in ok)
    e2e_list = sorted(r["e2e_ms"] for r in ok)
    tok = sum(r["tokens"] for r in ok)

    def pct(data, p):
        return data[int(len(data) * p / 100)] if data else 0

    slo_300 = sum(1 for t in ttft_list if t <= 300)
    slo_500 = sum(1 for t in ttft_list if t <= 500)

    summary = {
        "strategy": strategy, "rate": rate, "seed": SEED, "num_prompts": N,
        "completed": len(ok), "failed": len(results) - len(ok),
        "total_tokens": tok, "duration_s": elapsed,
        "throughput_tok_s": tok / elapsed if elapsed else 0,
        "ttft_p50": pct(ttft_list, 50), "ttft_p95": pct(ttft_list, 95),
        "tpot_p50": pct(tpot_list, 50), "tpot_p95": pct(tpot_list, 95),
        "e2e_p50": pct(e2e_list, 50), "e2e_p95": pct(e2e_list, 95),
        "slo300_pct": slo_300 / len(ok) * 100 if ok else 0,
        "slo500_pct": slo_500 / len(ok) * 100 if ok else 0,
    }

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{strategy}_seed{SEED}_rate{rate}.json")
    with open(fname, "w") as f:
        json.dump({"summary": summary}, f)

    print(f"[{strategy}] {len(ok)}/{N} ok, tp={summary['throughput_tok_s']:.0f}t/s")
    print(f"     TTFT P50={summary['ttft_p50']:.0f}ms P95={summary['ttft_p95']:.0f}ms")
    print(f"     TPOT P50={summary['tpot_p50']:.0f}ms P95={summary['tpot_p95']:.0f}ms")
    print(f"     SLO300={summary['slo300_pct']:.1f}% SLO500={summary['slo500_pct']:.1f}%")
    print(f"     Failed: {len(results) - len(ok)}")
    return summary


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "pe"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "results_poisson_v3"
    rate = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    asyncio.run(bench(strategy, port, out_dir, rate, max_tokens))

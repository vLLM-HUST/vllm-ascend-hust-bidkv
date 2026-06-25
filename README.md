# vllm-hust-ascend-bidkv

**BidKV: Utility-Guided Preemption Scheduling for KV-Pressure LLM Serving**

> 📄 **Paper**: accepted at **SC26** (Supercomputing 2026)
>
> BidKV is a utility-guided KV-cache reclamation policy that replaces implicit
> order-based victim selection with an explicit bid-based scheduling interface.
> It is evaluated on both vLLM and SGLang with Llama-3.1-8B-Instruct on NVIDIA RTX A6000
> under ShareGPT workloads.

---

## What BidKV Does

When the KV cache exceeds GPU memory during LLM serving, the engine must
**reclaim** KV state from active requests to admit new ones.  *Which* request
gets evicted (the **victim**) directly determines admission latency (TTFT) and
SLO attainment — yet existing systems rely on coarse order-based heuristics
(LIFO, FCFS) that ignore per-request reclamation cost.

**BidKV** turns victim selection into a structured auction:

1. Each active request submits a **bid** encoding:
   - `r` — recoverable KV capacity (tokens freed)
   - `δ` — estimated disruption cost (completion progress, preemption history)

2. The scheduler ranks bids by **utility** _U = r / (δ + ε)_ and preempts the
   highest-utility victim via the framework's native preemption path.

This is a **non-invasive scheduling layer**: BidKV controls *which* request is
reclaimed, not *how* — it delegates all reclamation to the framework's existing
mechanism, leaving output correctness intact.

---

## Key Results (from SC26 Paper)

| Metric | vLLM + BidKV vs. vLLM Native (PE) |
|---|---|
| TTFT P95 (cross-rate avg.) | **544 ms** (best among all strategies) |
| SLO@300ms improvement | **+14.8 pp** at rate=3.8 |
| Reclamation efficiency | Fewest events (181), highest freed-per-event (1,398 tok/evt) |
| Throughput tradeoff | ~7% lower than highest-throughput baseline |

On **SGLang**, the same unmodified policy achieves **+38.8 pp SLO** and
**18.4× TTFT P95 reduction** vs. SGLang's native LRU eviction, confirming
cross-framework portability.

---

## Architecture

BidKV is organized into four layers:

```
┌─────────────────────────────────┐
│  Runtime Adapter Layer          │  ← framework hooks (vLLM / SGLang)
├─────────────────────────────────┤
│  Utility-Ranked Selection Layer │  ← greedy bid solver
├─────────────────────────────────┤
│  Bid Generation Layer           │  ← δ = 1 + w_c·c + w_p·P
├─────────────────────────────────┤
│  Bid Signal Layer               │  ← U = r / (δ + ε)
└─────────────────────────────────┘
```

- **Scorer-agnostic**: the disruption estimator `δ` is pluggable; the current
  instantiation uses request-lifecycle features (completion ratio, preemption
  count, prompt length)
- **Framework-portable**: integrates with vLLM and SGLang via a minimal
  `FrameworkAdapter` ABC — no source-code modifications required

---

## Installation

```bash
pip install vllm-hust-ascend-bidkv
```

The plugin auto-registers via the `vllm_ascend.victim_selector` entry-point
and is discovered by vllm-ascend-hust at scheduler init time.

---

## Quick Start

```bash
# Enable utility-based victim selection
export VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION=1

# Optional: only activate reordering when KV utilization > 95%
export VLLM_ASCEND_UTILITY_KV_GATE=0.95

# Launch as usual
vllm serve /path/to/model ...
```

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION` | `0` | Enable BidKV |
| `VLLM_ASCEND_UTILITY_KILL_SWITCH` | `0` | Emergency bypass (1 = disable) |
| `VLLM_ASCEND_UTILITY_COMPLETION_WEIGHT` | `0.5` | Weight `w_c` for completion ratio penalty |
| `VLLM_ASCEND_UTILITY_PREEMPT_WEIGHT` | `0.3` | Weight `w_P` for starvation (preemption count) penalty |
| `VLLM_ASCEND_UTILITY_KV_GATE` | `0.0` | Min KV utilization fraction to enable reordering |
| `VLLM_ASCEND_UTILITY_COOLDOWN_S` | `0.0` | Cooldown (seconds) between utility-based selections |
| `VLLM_ASCEND_UTILITY_MIN_RUNNING` | `1` | Min running queue size to activate |
| `VLLM_ASCEND_UTILITY_SNAPSHOT_ENABLED` | `0` | Capture decision snapshots for debugging |
| `VLLM_ASCEND_UTILITY_SNAPSHOT_TOP_K` | `3` | Top-K candidates in snapshots |
| `VLLM_ASCEND_UTILITY_SNAPSHOT_HISTORY_SIZE` | `32` | Snapshot ring buffer capacity |

---

## Paper Reference

```bibtex
@inproceedings{bidkv2026,
  title     = {BidKV: Utility-Guided Preemption Scheduling for
               KV-Pressure LLM Serving},
  booktitle = {Proceedings of the International Conference for High
               Performance Computing, Networking, Storage, and Analysis (SC)},
  year      = {2026},
}
```

## License

Apache-2.0

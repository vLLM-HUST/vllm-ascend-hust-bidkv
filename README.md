# vllm-hust-ascend-bidkv

BidKV: Utility-based victim selection plugin for vLLM Ascend.

## Overview

When KV cache pressure triggers preemption, the default vLLM scheduler picks
the victim by simple FCFS (last-in) or PRIORITY policy.  BidKV replaces this
with a **utility-based ranking**:

```
U = r / (delta + epsilon)
```

where:
- `r` = tokens freed (num_computed_tokens)
- `delta = 1 + w_c * completion + w_p * preemptions`
- `epsilon` = small constant (1e-6)

Higher U means the request frees many tokens but is close to completion and
hasn't been repeatedly preempted — the ideal preemption candidate.

## Installation

```bash
pip install vllm-hust-ascend-bidkv
```

The plugin auto-registers via the `vllm_ascend.victim_selector` entry-point
and is discovered by vllm-ascend-hust at scheduler init time.

## Quick Start

```bash
# Enable utility-based victim selection
export VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION=1

# Optional: only enable when KV utilization > 80%
export VLLM_ASCEND_UTILITY_KV_GATE=0.8

# Launch as usual
vllm serve ...
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION` | `0` | Enable BidKV |
| `VLLM_ASCEND_UTILITY_KILL_SWITCH` | `0` | Emergency kill switch |
| `VLLM_ASCEND_UTILITY_COMPLETION_WEIGHT` | `0.5` | Weight for completion factor |
| `VLLM_ASCEND_UTILITY_PREEMPT_WEIGHT` | `0.3` | Weight for preemption count |
| `VLLM_ASCEND_UTILITY_KV_GATE` | `0.0` | Min KV utilization to enable |
| `VLLM_ASCEND_UTILITY_COOLDOWN_S` | `0.0` | Cooldown between utility picks |
| `VLLM_ASCEND_UTILITY_MIN_RUNNING` | `1` | Min running queue size |
| `VLLM_ASCEND_UTILITY_SNAPSHOT_ENABLED` | `0` | Capture decision snapshots |
| `VLLM_ASCEND_UTILITY_SNAPSHOT_TOP_K` | `3` | Top-K in snapshots |
| `VLLM_ASCEND_UTILITY_SNAPSHOT_HISTORY_SIZE` | `32` | Snapshot ring buffer size |

## License

Apache-2.0

# AGENTS.md

## Architecture

This is a **plugin** for `vllm-hust`, not a standalone application. It is auto-discovered via the `vllm.victim_selector` entry-point declared in `pyproject.toml`. The host vllm-hust scheduler calls `BidkvVictimSelector.from_vllm_config()` followed by `pick_victim()` on every preemption decision.

- **Entry point:** `vllm_ascend_bidkv/__init__.py` → `BidkvVictimSelector`
- **Core logic:** `vllm_ascend_bidkv/selector.py` (721 lines, config + utility ranking + metrics)
- **5 strategies:** `"pe"`, `"pe-sjf"`, `"static-random"`, `"largest-first"`, `"bidkv"`

## Dependencies

`pyproject.toml` declares `dependencies = []`. All runtime imports (`vllm.v1.core.sched.request_queue`, `vllm.v1.request`) come from the host `vllm-hust` installation. Tests and experiments also need `numpy` and `aiohttp` at runtime (not declared, expected in the host environment).

## Configuration

Config values are resolved with this priority: **`additional_config` dict > environment variables > defaults.**

### Via `additional_config` (passed to `vllm serve --additional-config`)
Keys use `snake_case` matching the dataclass field names:
- `enable_utility_victim_selection` (bool)
- `utility_strategy` (str)
- `utility_kill_switch`, `utility_completion_weight`, `utility_preempt_weight`
- `utility_kv_gate`, `utility_cooldown_s`, `utility_min_running`
- `utility_snapshot_enabled`, `utility_snapshot_top_k`, `utility_snapshot_history_size`
- `utility_epsilon`, `utility_default_max_tokens`

### Via environment variables
Prefixed `VLLM_ASCEND_UTILITY_*` in uppercase (e.g. `VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION`, `VLLM_ASCEND_UTILITY_STRATEGY`). See `selector.py:78-109` for the full mapping.

## Commands

```bash
# Install in dev mode
pip install -e .

# Run all tests (requires vllm-hust installed in the same env)
pytest tests/ -v

# Run a single test
pytest tests/test_selector.py::TestBidkvVictimSelector::test_utility_mode_prefers_higher_u -v
```

There is no lint, formatter, or typecheck configured in this repo.

## Testing quirks

Tests use `types.SimpleNamespace` mocks instead of real vLLM `Request` objects. The `_make_request` helper in `tests/test_selector.py:11` builds these — pay attention to the field names (`arrival_time`, `num_computed_tokens`, `output_token_ids`, etc.) as they must match what `selector.py` accesses. If the upstream `Request` type changes field names, tests must be updated accordingly.

## Experiment scripts (experiments/)

**Require Ascend NPU hardware.** They assume:
- Model at `/data/shared_models/Qwen2.5-7B-Instruct`
- Dataset at `/data/shared_datasets/ShareGPT_V3_unfiltered_cleaned_split.json`
- Cached prompts at `/tmp/fixed_prompts.json` (auto-generated from the dataset)
- NPU tool `npu-smi` available for monitoring HBM usage
- `LD_LIBRARY_PATH` includes Ascend driver/toolkit/ATB libraries

The scripts start/kill `vllm serve` processes, poll `"Application startup complete"` in logs to detect readiness, and use `aiohttp` to send completions requests to `http://localhost:8000/v1/completions`.

## No CI, no Docker

There are no CI workflows, pre-commit hooks, or Dockerfiles in this repo. The only automation is manual experiment scripts.

# Asymmetric Expert Placement — Implementation Report

## Overview

Implemented asymmetric expert placement for DisagMoE's colocate (unified engine) mode. Each GPU can now hold a different number of MoE experts, with per-device configuration and weighted DP-attention routing.

## Test Setup

| Parameter | Value |
|---|---|
| Cluster | 3 nodes (sgpu0, sgpu2, sgpu3) × 2 L40S GPUs = 6 GPUs |
| Model | gptoss_120b (reduced: 4 layers, 8 experts, top_k=2) |
| Expert allocation | `[2, 2, 1, 1, 1, 1]` — sgpu0 gets 4 experts, sgpu2 and sgpu3 get 2 each |
| DP weights | `[2, 2, 1, 1, 1, 1]` — proportional to expert count |
| Benchmark | 50 Poisson requests, rate=5/s, input 64–128 tokens, output 128–256 tokens |

## Test Results

```
e2e_duration:      9.47s
req_throughput:    5 req/s
token_throughput:  964 tokens/s
req_latency_mean:  488ms
req_latency_median: 470ms
req_latency_p99:   698ms
itl_latency_mean:  3ms
itl_latency_median: 2ms
itl_latency_p99:   4ms
```

All 50 requests completed successfully (HTTP 200). Advanced logging was enabled.

## Changes Made

### C++ (3 files)

| File | Change |
|---|---|
| `csrc/include/datatypes.hpp` | Added `n_total_experts` field to `ParallelConfig` struct (default 0 = backward compatible) |
| `csrc/muhelper/dispatcher.cpp` | `UnifiedDispatcher` now uses `n_total_experts` when set, instead of `ep * n_exp_per_rank`, to size `expert_to_rank` |
| `csrc/bindings.cpp` | Exposed `n_total_experts` in pybind11 binding |

### Python — Data Plumbing (4 files)

| File | Change |
|---|---|
| `disagmoe/utils/placement.py` | Added `local_expert_counts` dict to `ModelPlacement`; added `unique_expert_ids_at()` and `local_num_experts_at()` helpers; populated in `ColocatePlacement._solve()` |
| `disagmoe/frontend/ray_helper.py` | Added `local_num_experts` and `local_expert_ids` fields to `InitCoreArgs` |
| `disagmoe/frontend/controller.py` | Passes per-device expert info in `InitCoreArgs`; supports `attn_dp_weights` and `per_device_config` parameters; creates `DPSchedulerWeighted` when weights are provided; per-device `EngineConfig` overrides |
| `disagmoe/frontend/engine.py` | Stores `local_num_experts`/`local_expert_ids` from `InitCoreArgs`; builds asymmetric-aware expert rank mapping in `build_expert_executor()`; passes `n_total_experts` and per-device `n_exp_per_rank` to C++ |

### Python — Runtime (3 files)

| File | Change |
|---|---|
| `disagmoe/executor/executor.py` | `ExpertsExecutor` derives `local_num_experts` from `len(local_to_global_expert_rank)` and uses it for all tensor shapes and operator construction |
| `disagmoe/executor/cuda_graph.py` | `CUDAGraphExpertsExecutor` uses `local_num_experts` for static buffer shapes and CUDA graph capture |
| `disagmoe/scheduler/dpscheduler.py` | Added `DPSchedulerWeighted` class; updated `get_dp_scheduler()` to accept optional weights |

### Python — Config Parsing (1 file)

| File | Change |
|---|---|
| `benchmark/benchmark_serving.py` | `resolve_expert_allocation()` now returns `(expert_allocation, attn_dp_weights, per_device_config)` tuple; passes all three to controller |

### Bugfix (1 file)

| File | Change |
|---|---|
| `disagmoe/frontend/tokenizer.py` | Fixed pre-existing ZeroDivisionError in `log_throughput()` that crashed the detokenizer thread |

### Experiment Infrastructure (3 new files)

| File | Purpose |
|---|---|
| `experiments/scripts/sphere-3/asym_expert_alloc.json` | Allocation config: 2:2:1:1:1:1 experts + DP weights |
| `experiments/scripts/sphere-3/launch_server.sh` | Launch script for 3-node asymmetric test (4 layers, 8 experts, top_k=2) |
| `experiments/scripts/sphere-3/README.txt` | Setup and usage instructions |

## Design Decisions

1. **Backward compatibility**: All changes are gated. When `local_expert_ids` is `None` or empty, the engine falls back to the original uniform `num_experts_per_rank` formula. When `n_total_experts` is 0 in C++, the old `ep * n_exp_per_rank` formula is used.

2. **Decoupling**: New logic (weighted scheduler, config parsing extension, per-device overrides) is in new classes/functions. Existing code paths are unchanged when asymmetric mode is disabled.

3. **C++ minimal change**: Only the `expert_to_rank` sizing in `UnifiedDispatcher` needed fixing. All other C++ dispatch logic (muhelper, channels) already uses `expert_ranks` tuples directly, which work correctly with asymmetric placement.

4. **Advanced logging**: No changes needed — `AdvancedLogger` instruments per-step execution time and queuing delay, which naturally works with asymmetric expert counts.

## JSON Config Format

```json
{
  "allocations": [
    {"host_ip": "10.0.0.1", "cuda_device": "0", "num_experts": 2},
    ...
  ],
  "attn_dp_weights": {
    "0": 2.0, "1": 2.0, "2": 1.0, ...
  },
  "per_device_config": {
    "0": {"max_batch_size_expert": 768}
  }
}
```

- `allocations`: Required. Per-device expert count.
- `attn_dp_weights`: Optional. DP rank → weight. Missing = equal weights (backward compatible).
- `per_device_config`: Optional. Per-device `EngineConfig` field overrides.

# Design: Asymmetric Expert Placement + Weighted DP-Attention Routing

**Scope**: Colocate mode only. Pipeline/interleave modes are out of scope.

---

## 1. Current State

### 1a. Expert Placement (partially done)

The asymmetric expert allocation feature **already exists** for placement:
- `launch_server.sh`: `ENABLE_ASYMMETRIC_DEPLOYMENT=1` + JSON config file
- `benchmark_serving.py:resolve_expert_allocation()`: Parses JSON, maps `(host_ip, cuda_device)` → `device_id`, produces `List[int]` of expert counts per device
- `ColocatePlacement._solve()` (placement.py:473-483): Assigns experts sequentially by device — GPU 0 gets experts 0..N₀-1, GPU 1 gets N₀..N₀+N₁-1, etc.
- `_update_expert_rank()`: Builds `expert_ranks` dict from actual device assignments

**What's broken**: The global `ModelConfig.num_experts_per_rank` property (config.py:34) returns `num_experts // ep_size` — a single global value. With asymmetric allocation (e.g., 3 experts on GPU 0, 1 expert on GPU 2), this value is meaningless. It's used in ~20 places across Python and C++:

| Location | Usage | Problem |
|---|---|---|
| `engine.py:377-381` | `build_expert_executor()` — builds local↔global expert rank mapping | Mapping is `[rank*N_per_rank..(rank+1)*N_per_rank)` which is wrong when counts differ |
| `engine.py:589` | Passed to C++ `ParallelConfig.n_exp_per_rank` | C++ dispatchers use this to compute inner expert ranks |
| `executor.py:354,360,362,386,396,464` | Expert executor: `expert_ids`, `batch_sizes` tensor shapes, GEMM dispatch | Tensors sized to global `num_experts_per_rank` — must match actual local count |
| `cuda_graph.py:262,272,335,390` | CUDA graph expert executor: same as above | Same problem |
| `models/utils.py:182-195` | `make_dummy_expert_input()` — test/warmup utility | Divides batch evenly by `num_experts_per_rank` |
| C++ `muhelper.cpp` | `cfg.n_exp_per_rank` used in dispatcher inner_expert_ranks | Wrong mapping for asymmetric |

### 1b. DP-Attention Routing (not started)

Requests are routed to attention DP replicas by `DPScheduler` (disagmoe/scheduler/dpscheduler.py):
- **`DPSchedulerMax`** (production default): Picks rank with most free KV-cache blocks. Greedy, no weights.
- **`DPSchedulerRR`**: Strict round-robin. No weights.

Neither has any weight parameter. The scheduling decision happens in `DPScheduler.schedule()` → `_schedule(seq_len) → rank`. The chosen rank is passed to `tokenizer.put_single_request(req_id, prefill_len, output_len, dp_rank=rank)` which pushes to a per-rank ZMQ socket.

---

## 2. Changes Required

### 2a. Per-Device Expert Count (fix `num_experts_per_rank`)

**Goal**: Each GPU knows its own local expert count. No global `num_experts_per_rank` assumption.

#### Step 1: Add per-device expert count to `ModelPlacement`

```python
# placement.py — ModelPlacement dataclass
local_expert_counts: Dict[int, int] = None  # device_id → number of experts on this device
```

Populated in `ColocatePlacement._solve()`:
- If `expert_allocation` is provided: `local_expert_counts[dev_id] = expert_allocation[dev_id]`
- If not (uniform): `local_expert_counts[dev_id] = num_experts // ep_size` for each expert device

#### Step 2: Propagate to each worker via `InitCoreArgs`

```python
# ray_helper.py — InitCoreArgs
local_num_experts: int = 0  # how many experts THIS device holds
```

Set in `controller.py:init_engine()`:
```python
local_num_experts=len(model_place.expert_ids_at(device_id)) // model_config.num_layers
```
(Each expert appears once per layer in colocate mode, so dividing by `num_layers` gives the actual expert count on this device.)

#### Step 3: Use local count in `engine.py:build_expert_executor()`

Replace all uses of `self.model_config.num_experts_per_rank` in the local rank mapping with `self.local_num_experts` (received from `InitCoreArgs`).

```python
# engine.py — build_expert_executor()
# BEFORE:
self.local_to_global_expert_rank = [0] * self.model_config.num_experts_per_rank
for i in range(self.model_config.num_experts_per_rank):
    self.local_to_global_expert_rank[i] = self.model_config.num_experts_per_rank * self.rank_in_group + i

# AFTER: use actual expert IDs from placement
# self.local_expert_ids is derived from model_place.expert_ids_at(device_id)
# e.g., if this GPU holds experts [3, 4, 5], then:
#   local_to_global = [3, 4, 5]
#   global_to_local = {3: 0, 4: 1, 5: 2}
```

The local expert IDs are already available — `model_place.expert[device_id]` gives `[(layer_id, expert_id), ...]`. Extract the unique `expert_id`s.

#### Step 4: Pass per-device count to C++ `ParallelConfig`

```python
# engine.py:585-591
ParallelConfig.to_c(
    1,
    self.model_config.ep_size,
    self.model_config.dp_size,
    self.local_num_experts,  # was: self.model_config.num_experts_per_rank
    core_args.expert_ranks,
)
```

#### Step 5: Update executor tensor shapes

In `executor.py` and `cuda_graph.py`, replace `self.model_config.num_experts_per_rank` with `self.local_num_experts` (or `len(self.local_to_global_expert_rank)`) for:
- `self.expert_ids` tensor (arange)
- `self.static_input_batch_sizes` shape
- `self.batch_sizes_gdr` buffer
- GEMM dispatch `num_experts_per_rank` argument
- `make_dummy_expert_input()` calls

#### Summary of touched files

| File | Change |
|---|---|
| `disagmoe/utils/placement.py` | Add `local_expert_counts` to `ModelPlacement`, populate in `ColocatePlacement._solve()` |
| `disagmoe/frontend/ray_helper.py` | Add `local_num_experts` and `local_expert_ids` to `InitCoreArgs` |
| `disagmoe/frontend/controller.py` | Compute and pass `local_num_experts` / `local_expert_ids` per worker |
| `disagmoe/frontend/engine.py` | Use `local_num_experts` instead of global `num_experts_per_rank` in `build_expert_executor()` and `init_core()` |
| `disagmoe/executor/executor.py` | Replace `model_config.num_experts_per_rank` with local count for tensor shapes |
| `disagmoe/executor/cuda_graph.py` | Same as executor.py |
| `disagmoe/models/utils.py` | `make_dummy_expert_input()` takes explicit count instead of global config |

**NOT touched**: `disagmoe/config.py` — `ModelConfig.num_experts_per_rank` property stays as-is (still useful as a default / for uniform case). We just stop using it as the sole source of truth on per-device paths.

---

### 2b. Weighted DP-Attention Routing

**Goal**: Allow specifying a weight per DP-attention rank. Higher weight = proportionally more requests routed there.

#### Config Format

Extend the existing JSON config file (or use the same `--expert-allocation-path` file):

```json
{
  "allocations": [
    { "host_ip": "10.0.0.1", "cuda_device": "0", "num_experts": 3 },
    { "host_ip": "10.0.0.1", "cuda_device": "1", "num_experts": 1 },
    ...
  ],
  "attn_dp_weights": {
    "0": 2.0,
    "1": 1.0
  }
}
```

`attn_dp_weights` maps **DP rank** (string key, since JSON) → relative weight. Weights are normalized internally (2.0 and 1.0 → 2/3 and 1/3). If the key is absent, all ranks get equal weight (backward compatible).

#### New Scheduler: `DPSchedulerWeighted`

```python
# dpscheduler.py

class DPSchedulerWeighted(DPScheduler):
    """Capacity-aware scheduling with configurable weight bias."""

    def __init__(self, dp_size: int, block_size: int, weights: List[float]):
        super().__init__(dp_size, block_size)
        assert len(weights) == dp_size
        total = sum(weights)
        self.weights = [w / total for w in weights]  # normalized to sum=1

    @override
    def _schedule(self, seq_len: int) -> int:
        required = self.required_blocks(seq_len)
        best_rank = -1
        best_score = -1.0
        for i, num_blocks in enumerate(self.kv_cache_stats):
            if num_blocks < required:
                continue
            # Score = available_blocks * weight
            # Higher weight → more attractive even with fewer free blocks
            score = num_blocks * self.weights[i]
            if score > best_score:
                best_score = score
                best_rank = i
        return best_rank
```

The scoring multiplies available capacity by the configured weight. A rank with weight 2.0 (normalized to 2/3) is twice as attractive as a rank with weight 1.0 (normalized to 1/3) at equal capacity. As a rank fills up, its `num_blocks` drops naturally, creating backpressure.

**Why not pure weighted-random?** Pure weighted-random ignores capacity — it'd keep sending 2/3 to rank 0 even when rank 0 is out of KV-cache blocks. The multiplicative approach respects both the weight preference AND available capacity.

#### Plumbing

1. **`benchmark_serving.py:resolve_expert_allocation()`** — also parse `attn_dp_weights` from the JSON, return it alongside expert allocation
2. **`benchmark_serving.py:launch()`** — pass weights to `get_dp_scheduler()`
3. **`controller.py:init_engine()`** — store weights, pass to `get_dp_scheduler(dp_size, block_size, "weighted", weights=weights)`
4. **`dpscheduler.py`** — add `DPSchedulerWeighted` to `_clses` dict, update `get_dp_scheduler()` signature to accept optional `weights`
5. **`launch_server.sh`** — no new macro needed; weights live in the same JSON config file gated by `ENABLE_ASYMMETRIC_DEPLOYMENT`

#### Summary of touched files

| File | Change |
|---|---|
| `disagmoe/scheduler/dpscheduler.py` | Add `DPSchedulerWeighted`, update `get_dp_scheduler()` |
| `benchmark/benchmark_serving.py` | Parse `attn_dp_weights` from JSON, pass through |
| `disagmoe/frontend/controller.py` | Accept and forward weights to scheduler factory |

---

## 3. Config File: Unified Example

```json
{
  "allocations": [
    { "host_ip": "10.0.0.1", "cuda_device": "0", "num_experts": 3 },
    { "host_ip": "10.0.0.1", "cuda_device": "1", "num_experts": 1 },
    { "host_ip": "10.0.0.2", "cuda_device": "0", "num_experts": 2 },
    { "host_ip": "10.0.0.2", "cuda_device": "1", "num_experts": 2 }
  ],
  "attn_dp_weights": {
    "0": 3.0,
    "1": 1.0,
    "2": 2.0,
    "3": 2.0
  }
}
```

This means: GPU 0 on node 1 gets 3 experts + 3/8 of attention requests. GPU 1 on node 1 gets 1 expert + 1/8 of attention requests. Etc.

---

## 4. Validation Criteria

1. **Uniform allocation still works**: When `expert_allocation` is `None` (no JSON config), everything must behave identically to current code
2. **Expert count sum**: `sum(all num_experts) == model_config.num_experts` (existing assertion)
3. **Weight sum**: Weights must all be > 0. Missing `attn_dp_weights` → equal weights (1.0 each)
4. **Tensor shape correctness**: Expert executor tensor shapes (expert_ids, batch_sizes) match actual local expert count, not global `num_experts_per_rank`
5. **CUDA graph replay**: CUDA graph expert executor must be re-captured with correct local expert count
6. **C++ n_exp_per_rank**: Each GPU's `ParallelConfig.n_exp_per_rank` matches its actual local count
7. **End-to-end**: Warmup + stress test with asymmetric config (e.g., [3,1,2,2]) must pass

---

## 5. Risk / Open Questions

1. **CUDA graph batch size buckets**: The expert CUDA graph captures fixed batch-size buckets. With fewer experts on a GPU, each expert may receive larger batches. Need to verify `max_batch_size_expert` is still sufficient, or make it configurable per-device.

2. **Expert weight loading**: The model weight loading code needs to load only the correct subset of expert weights per device. Currently done via `local_to_global_expert_rank` mapping — as long as this mapping is correct (Step 3 above), weight loading should work. Needs verification.

3. **C++ dispatcher `inner_expert_ranks`**: The C++ side uses `cfg.n_exp_per_rank` to build the mapping from global expert ID to inner (local) expert index. With per-device counts, each device's `n_exp_per_rank` is already passed independently (Step 4). But need to verify the C++ mapping logic handles non-contiguous or unevenly-sized expert assignments correctly.

4. **Weighted scheduler convergence**: Under extreme weight skew (e.g., 99:1), the heavily-weighted rank fills up fast, and the capacity term dominates, pushing requests to the lightly-weighted rank anyway. This is by design (prevents OOM), but users should understand that weights are *preferences*, not guarantees.

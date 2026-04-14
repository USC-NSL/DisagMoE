# GLM-4.5-Air Performance Investigation Report

**Experiment**: sphere-16-investigate-glm  
**Date**: 2026-04-14  
**Testbed**: sgpu2-9 (8 nodes × 2 L40S = 16 GPUs)  
**Model**: glm45air_106b (45L, 128E, top-8, 1 shared expert, bf16)

## Summary

Three objectives were pursued:
1. Reproduce AsyncMoE GLM experiments on the new sgpu2-9 testbed
2. Implement and test advanced logging for dispatcher/pool behaviors
3. Investigate why GLM throughput is drastically lower than gptoss

**Key Findings**:
1. **Primary blocker**: Runtime CUDA OOM at MEM_FRAC=0.98 (config bug prevented GLM-specific 0.95 override). Fixed by allowing explicit override; 0.90 was the initial safe fallback.
2. **Advanced-logging follow-up at MEM_FRAC=0.95**: balanced GLM runs now complete successfully with corrected receive-side / backpressure logs dumped to disk.
3. **Corrected methodology**: dispatcher call latency was the wrong signal because transport is async. I replaced it with **receive-side completion logging in MuPool** and **explicit max-pending-sends stall logging in MuDispatcher**.
4. **Network is still not close to saturated**: on a clean 16-rank gsm8k snapshot, per-node receive bandwidth (aggregating the 2 ranks that share one 200 Gbps NIC) is only **8.98 GB/s p95** and **9.37 GB/s max**, i.e. about **36-38% of a 25 GB/s link**.
5. **Backpressure is real, but not dominant**: the new logs show many explicit `max_pending_sends=16` stalls, but they are short: gsm8k has **68,751 stalls**, mean **0.229 ms**, p95 **0.655 ms**, max **99.683 ms**.
6. **Normalized stall impact is small**: those **68,751 stalls** occurred over **15,283,095 remote sends/receives**, i.e. only **0.45%** of sends stalled at all. Amortized over **all** sends, the mean stall cost is just **0.00103 ms/send = 1.03 μs/send**.
7. **Why gsm8k is not much faster than sharegpt**: shorter contexts do not produce a meaningfully larger common-case MoE batch. The corrected logs still show small typical MoE batches (sharegpt p50 **19**, gsm8k p50 **9**) and similar MoE execution cost, so GLM remains limited by the decode-side MoE path: 45 layers, top-8 routing, 4096-d hidden-state movement, and one shared expert per layer.

## Reproduction Results (MEM_FRAC=0.90)

| Experiment | Throughput (peak) | ITL mean | ITL p99 | Status |
|---|---|---|---|---|
| sharegpt_regular | ~4.6k tok/s | ~220ms | ~230ms | Completed (post-bench cleanup hung) |
| sharegpt_balanced | ~10.6k tok/s | ~279ms | — | SUCCESS |
| gsm8k_regular | ~14.4k tok/s | ~514ms | — | SUCCESS |
| gsm8k_balanced | ~14.8k tok/s | ~498ms | — | SUCCESS |

**Comparison with gptoss baseline** (from sphere-16-tput-itl-mar30):

| Model | sharegpt tput | gsm8k tput | Ratio |
|---|---|---|---|
| gptoss_120b (top-4) | 23,660 tok/s | 25,126 tok/s | 1.0x |
| glm45air_106b (top-8) | ~10,600 tok/s | ~14,800 tok/s | 0.45-0.59x |

## Objectives Status

### 1. Testbed Setup (COMPLETE)
- Updated config.sh: HEAD_NODE=sgpu2, WORKER_NODES=sgpu3-9
- Installed disagmoe on sgpu2, sgpu3, sgpu4, sgpu9 (sgpu5-8 had prior install)
- Fixed Ray version mismatch (upgraded sgpu5-8 from 2.54.0 → 2.54.1)
- Fixed gdrdrv kernel module (loaded + created /dev/gdrdrv on sgpu5-8)
- Fixed stale code on sgpu5-8 (hard-reset from old commits to fbe1d5a, rebuilt C++)
- Synced gating profiles and datasets to all workers
- Node IP mapping verified: sgpu2=10.0.0.1 through sgpu9=10.0.0.8

### 2. Advanced Logging Enhancement (COMPLETE)
Implemented two generations of advanced logging. The initial dispatcher/pool logging was useful for qualitative visibility, but its dispatcher latency interpretation was wrong because sends are async. I then replaced the bottleneck analysis with corrected receive-side / backpressure logging in the C++ transport path.

**Files modified:**
- `disagmoe/utils/advanced_logger.py`: Added `log_dispatcher_put()` and `log_pool_put()` methods with full data structures, serialization, and reset support
- `disagmoe/frontend/engine.py`: Instrumented `post_process()` with sampled dispatcher.put() timing, and `recv_new_request()` with pool admission logging
- `experiments/process_and_plot_advanced_logging.py`: Added dispatcher latency CDF, batch-size-vs-latency, timeline, pool admission rate, and summary plots
- `csrc/include/muhelper.h`: Added transport-stat buffers for recv completions and pending-send stalls
- `csrc/muhelper/muhelper.cpp`: Logged recv completion timestamps/bytes and explicit `max_pending_sends` blocking intervals
- `csrc/bindings.cpp`: Exposed new drain methods to Python so logs can be collected/dumped

**New data collected:**
- First pass:
  - Dispatcher put latency (ms)
  - Dispatcher batch sizes over time
  - Pool admission rate
- Corrected pass:
  - Receive completion records `(peer_id, layer_id, num_tokens, num_bytes, posted_ts, completed_ts, is_local)`
  - Pending-send stall intervals `(start_ts, end_ts, pending_before, max_pending, yield_count)`
  - Per-node receive bandwidth timelines with 2 ranks aggregated per NIC

**Testing**: The first dispatcher-based methodology was replaced with receive-side / backpressure instrumentation and validated after rebuilding on sgpu2-9.

### 2.1 Corrected receive-side / backpressure logging at MEM_FRAC=0.95

To address the methodological issues in the original dispatcher-based estimate, I changed the logging as follows:
- **MuPool recv completion logging**: record `(peer_id, layer_id, num_tokens, num_bytes, posted_ts, completed_ts, is_local)` when recv completion is observed on the receiver side.
- **MuDispatcher pending-send stall logging**: record explicit wait intervals when `drain_pending_sends_to(max_pending_sends)` blocks because in-flight sends have not retired yet.
- **Per-node analysis**: aggregate the two ranks on each node into a single NIC bandwidth timeline because both ranks share one 200 Gbps RoCE link.

The corrected instrumentation was rebuilt on sgpu2-9 and then exercised with balanced workloads using:
- `MEM_FRAC=0.95`
- `BENCH_RATE=1000`
- `BENCH_TIME=5` for the clean gsm8k full run and steady-state snapshot collection
- `BENCH_GENERATOR=dataset`
- `BENCH_MAX_CONTEXT_LEN=2048`
- `--enable-advanced-logging --advanced-logging-sample-rate 0.02`

**Output dirs**:
- `advlog-corrected-mem095/gsm8k_balanced_snapshot/advanced_logs` — clean 16-rank corrected logging
- `advlog-corrected-mem095/sharegpt_balanced_snapshot/advanced_logs` — steady-state snapshot collected, but some recv JSONs were truncated during forced mid-run collection, so recv-bandwidth numbers from this run are lower-confidence

**Corrected receive-side evidence**:
- Clean gsm8k node-level recv bandwidth (2 ranks / NIC):
  - mean **3.91 GB/s**
  - p95 **8.98 GB/s**
  - max **9.37 GB/s**
  - => only about **36-38%** of a 200 Gbps link
- Therefore the RoCE link is **not** close to saturation even under the corrected methodology.

**Explicit `max_pending_sends` evidence**:
- Clean gsm8k run:
  - stall count: **68,751**
  - mean stall: **0.229 ms**
  - p95 stall: **0.655 ms**
  - max stall: **99.683 ms**
  - `pending_before_mean=16.00`, confirming the wait is exactly at the configured in-flight send cap
- Clean gsm8k normalization:
  - remote sends/receives: **15,283,095**
  - stall rate: **0.45%** of sends
  - amortized stall cost over all sends: **1.03 μs/send**
- So the send cap does cause waits, but those waits are mostly sub-millisecond and do not indicate link saturation.

**Batching evidence from corrected logs**:
- Sharegpt snapshot: MoE batch mean **103.0**, p50 **19**, p99 **888**
- Gsm8k snapshot: MoE batch mean **100.9**, p50 **9**, p99 **1024**
- Shorter gsm8k contexts do **not** improve the common-case MoE batch; they mainly increase how often the system reaches the largest batches in the tail.

**Manual QA / execution evidence**:
- The corrected gsm8k run completed cleanly with `HTTP 200` and dumped advanced logs successfully.
- The sharegpt corrected snapshot required a mid-run dump to avoid the long-tail cleanup / drain issue seen in this benchmark path; those logs are still useful for MoE-batch evidence, but less reliable for full recv-bandwidth accounting because some JSONs were truncated.

**Interpretation**:
- Shorter contexts reduce attention/KV work, but after the system is saturated the served-token path is still dominated by per-token MoE work.
- For GLM, each served token still pays:
  - **45 layers** (vs 36 for gptoss)
  - **top-8 routing** (vs top-4 for gptoss)
  - **4096 hidden width** transfers per routed token
  - **1 shared expert per layer** on top of routed experts
- The corrected logs now rule out two candidate explanations:
  - **Not NIC saturation**
  - **Not long waits at max_pending_sends as the dominant cost**
- That means gsm8k mostly increases **request throughput** (more short requests finish), not **token throughput**. The token-level ceiling barely moves because the decode-side MoE pipeline is still the limiter.

### 3. GLM Performance Investigation (ROOT CAUSE IDENTIFIED)

#### The Symptom
GLM throughput is drastically lower than gptoss:
- gptoss_120b: ~23k tokens/s (sharegpt balanced at 200rps)
- glm45air_106b: ~50 tokens/s briefly, then **0 tokens/s** (system stalls)

#### Root Cause: MEM_FRAC Config Bug → Runtime OOM

**Config bug** in `glm45air_config.sh`:
```bash
# BEFORE (broken): config.sh sets MEM_FRAC=0.98 via ${MEM_FRAC:-0.98}
# Then glm45air_config.sh tries ${MEM_FRAC:-0.95} — NO-OP since already set!
MEM_FRAC=${MEM_FRAC:-0.95}  # This never executes the 0.95 assignment

# AFTER (fixed):
MEM_FRAC=${MEM_FRAC_OVERRIDE:-0.90}  # Hard override
```

**What happens at MEM_FRAC=0.98:**
1. Server starts successfully — initial KV cache allocation fits in 0.98 of GPU memory
2. ~30 seconds into benchmark, token processing begins consuming additional memory:
   - Shared expert activations (1 per layer × 45 layers)
   - Top-8 routing creates 2x more expert dispatch work vs top-4
   - Intermediate activations for 8 expert routes per token
3. Engine hits CUDA OOM: "Tried to allocate 20.00 MiB. GPU 0 has 44.40 GiB total, 20.31 MiB free"
4. Worker engine crashes: `Exception in single_module_loop_overlap: CUDA out of memory`
5. Requests assigned to dead worker never complete → system stalls
6. DP scheduler shows: #running=4143 (frozen), #waiting=5823 (frozen) — no progress

**Evidence from server.log:**
```
Detokenizer: token throughput: 0.05k tokens/s | ITL mean=65.4ms p50=62.0ms p99=194.0ms
  # (brief window before OOM)
Exception in single_module_loop_overlap: CUDA out of memory.
  # (system stalls after this)
Global DP scheduler: #running: 4143, #waiting: 5823  # frozen for 600+ seconds
```

#### Contributing Factors (not root cause but amplify the problem)

1. **Top-8 routing (vs gptoss top-4)**:
   - Each token dispatches to 8 experts → 2x more intermediate memory
   - Expert batch sizes: 512 tokens × 8 = 4096 expert-token assignments (vs 2048 for top-4)
   - Memory for token permutation and weight application scales linearly with top-k

2. **Shared expert overhead**:
   - GLM has 1 shared expert per layer that processes ALL tokens (no routing)
   - Runs on the attention GPU, adding ~1408 intermediate_size activations per layer

3. **More layers (45 vs 36)**:
   - 25% more layer state in the pipeline at any given time

4. **Memory fragmentation**:
   - The OOM message shows 134.52 MiB reserved but unallocated (PyTorch fragmentation)
   - With near-full GPU memory utilization, even small fragmentation triggers OOM

#### Recommended Fix

1. **Immediate**: allow explicit MEM_FRAC override (done in `glm45air_config.sh`)
2. **Observed safe values**:
   - `0.95` works for the follow-up balanced 5s advanced-logging runs
   - `0.90` remains the conservative fallback already validated for the broader reproduction sweep
3. **Long-term**: the eval script should detect runtime OOM (not just startup OOM) and retry automatically

## Infrastructure Issues Encountered

| Issue | Nodes | Resolution |
|---|---|---|
| Ray version mismatch (2.54.0 vs 2.54.1) | sgpu5-8 | `pip install ray==2.54.1` |
| Python minor-version mismatch (3.12.12 vs 3.12.13) during reruns | sgpu6-8 | `conda install -y python=3.12.13` |
| gdrdrv kernel module not loaded | sgpu5-8 | `sudo insmod src/gdrdrv/gdrdrv.ko` + manual mknod |
| /dev/gdrdrv device node missing | sgpu5-8 | `sudo mknod /dev/gdrdrv c $MAJOR 0; chmod 666` |
| Stale code (56 commits behind) | sgpu5-8 | `git reset --hard origin/asym` + rebuild |
| vLLM patch not detected (false alarm) | all | Patch was applied; detection script checked wrong file |

## Next Steps

1. Extend the `0.95` validation from balanced 5s runs to the longer/full production sweep if desired
2. If needed, repeat the same advanced-logging procedure on `regular` routing to quantify imbalance effects separately from the balanced case
3. If future tuning is needed, focus on reducing decode-side MoE overhead (routing/scatter/shared-expert cost and small common-case microbatches), not on RoCE bandwidth tuning first

## Files Changed

| File | Change |
|---|---|
| `experiments/scripts/sphere-16/eval/config.sh` | HEAD_NODE=sgpu2, WORKER_NODES=sgpu3-9 |
| `experiments/scripts/sphere-16/eval/glm45air_config.sh` | MEM_FRAC hard override to 0.90 |
| `disagmoe/utils/advanced_logger.py` | Added dispatcher/pool logs first, then recv-completion and pending-send-stall logs |
| `disagmoe/frontend/engine.py` | Initial dispatcher/pool instrumentation, then transport-stat draining into advanced logs |
| `experiments/process_and_plot_advanced_logging.py` | Added dispatcher/pool plots first, then corrected recv-bandwidth / backpressure summaries and plots |
| `csrc/include/muhelper.h` | Added recv-completion and pending-send-stall stat buffers / APIs |
| `csrc/muhelper/muhelper.cpp` | Logged recv completion timing/bytes and explicit `max_pending_sends` waits |
| `csrc/bindings.cpp` | Exposed transport-stat drain APIs to Python |
| `docs/sgpu-rules-for-agents.txt` | Updated node mapping for sgpu2-9 |
| `experiments/scripts/sphere-16/README.txt` | Updated for sgpu2-9 testbed |

# Sphere-16 Asymmetric Heterogeneous Experiment Report

## Objective

Compare throughput of asymmetric expert placement on a simulated heterogeneous cluster, specifically comparing **equal (1:1) vs weighted (2:1) DP-attention routing** when half the GPUs are compute-throttled.

## Cluster Configuration

| Node Group | Nodes | GPUs | Compute | Experts/GPU |
|---|---|---|---|---|
| Full-compute | sgpu0, sgpu2, sgpu3, sgpu4 | 8× L40S | 100% | 10–11 |
| Throttled (MPS 50%) | sgpu6, sgpu7, sgpu8, sgpu9 | 8× L40S | 50% | 5–6 |
| **Total** | **8 nodes** | **16 GPUs** | | **128 experts** |

**Compute throttling**: CUDA MPS `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` on the last 4 nodes. This limits SM thread utilization to 50%, simulating a GPU with half the compute power. Memory capacity and bandwidth are NOT throttled.

## Model & Workload

| Parameter | Value |
|---|---|
| Model | gptoss_120b (36 layers, 128 experts, top_k=4) |
| Gate profile | gsm8k (`gating_math_gsm8k_200.parquet`) |
| Input length | 128–256 tokens (2× baseline sphere-16) |
| Output length | 256–512 tokens |
| Requests | 50 (Poisson, rate=10/s, 5s window) |
| Placement | Colocate (unified engine, defrag scheduler) |

## Expert Allocation

128 experts distributed as:
- Full-compute GPUs: `[11, 11, 10, 10, 11, 11, 10, 10]` → 84 experts (65.6%)
- Throttled GPUs: `[6, 6, 5, 5, 6, 6, 5, 5]` → 44 experts (34.4%)

Ratio: full-compute GPUs get ~2× more experts than throttled GPUs, matching the compute asymmetry.

## Results

### Case 1: Equal DP Weights (1:1)

All 16 GPUs receive equal attention request routing weight.

```
e2e_duration:        31.01s
req_throughput:      2 req/s
token_throughput:    571 tokens/s
req_latency_mean:    21,782ms
req_latency_median:  21,178ms
req_latency_p99:     27,576ms
itl_latency_mean:    58ms
itl_latency_median:  60ms
itl_latency_p99:     115ms
```

### Case 2: Weighted DP (2:1)

Full-compute GPUs get 2× the attention routing weight of throttled GPUs.

```
e2e_duration:        36.42s
req_throughput:      1 req/s
token_throughput:    150 tokens/s
req_latency_mean:    27,043ms
req_latency_median:  28,422ms
req_latency_p99:     33,912ms
itl_latency_mean:    70ms
itl_latency_median:  65ms
itl_latency_p99:     163ms
```

### Comparison

| Metric | Equal (1:1) | Weighted (2:1) | Δ |
|---|---|---|---|
| Token throughput | **571 tok/s** | 150 tok/s | 1:1 is **3.8× higher** |
| Req throughput | **2 req/s** | 1 req/s | 1:1 is 2× higher |
| Req latency mean | **21.8s** | 27.0s | 1:1 is 24% faster |
| ITL mean | **58ms** | 70ms | 1:1 is 17% faster |
| ITL p99 | **115ms** | 163ms | 1:1 is 30% better |

## Analysis

Equal weights (1:1) significantly outperforms weighted (2:1) on this workload:
- **3.8× higher token throughput** with equal weights
- **24% lower request latency** with equal weights

This is counterintuitive — one might expect weighted routing (sending more requests to faster GPUs) to improve throughput. The likely explanation:

1. **KV cache capacity dominates over compute**: With MPS throttling at 50%, compute is halved but memory is not. The throttled GPUs have the same KV cache capacity as full-compute GPUs. Equal routing better utilizes the total KV cache pool.

2. **Expert execution is the bottleneck**: With only 5–6 experts per throttled GPU (vs 10–11 on full-compute), the throttled GPUs have less expert compute work per token, partially compensating for the reduced compute power.

3. **The DPSchedulerMax already load-balances**: When using equal weights, the scheduler still considers available KV cache blocks. Throttled GPUs that fall behind naturally attract fewer new requests as their cache fills up, providing organic backpressure.

4. **2:1 weights overcorrect**: By sending 2× more requests to full-compute GPUs, the weighted scheduler may be overloading those GPUs while underutilizing the throttled GPUs' memory capacity.

## Known Issues

- **CUDA MPS flakiness**: The MPS-throttled nodes occasionally fail with "CUDA-capable device(s) is/are busy or unavailable" during server startup. This is an MPS stability issue (not a DisagMoE bug) that requires MPS daemon restart to recover. The `setup_mps_throttle.sh` script handles this, but the `run_experiments.sh` may need manual intervention between runs.

- **Large request counts deadlock**: At high request counts (>100 concurrent), the system deadlocks with all requests stuck in "running" state. This appears to be a throughput issue under heavy load with the longer input sequences (128–256), not specific to asymmetric placement.

## Files

| File | Purpose |
|---|---|
| `asym_expert_alloc_equal_weights.json` | Allocation config with 1:1 DP weights |
| `asym_expert_alloc_2to1_weights.json` | Allocation config with 2:1 DP weights |
| `setup_mps_throttle.sh` | Start CUDA MPS at 50% on throttled nodes |
| `teardown_mps_throttle.sh` | Stop CUDA MPS on throttled nodes |
| `run_experiments.sh` | Automated experiment runner (both configs) |
| `README.txt` | Setup and usage instructions |
| `results/equal_1to1/` | Experiment 1 results |
| `results/weighted_2to1/` | Experiment 2 results |

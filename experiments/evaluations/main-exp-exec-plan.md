# amoe-064 Experiment Execution Plan

## Overview

16 experiment runs across 4 system configurations × 2 profiles × 2 balanced variants.

### Systems
1. **sglang EP16** — EP16 with mem-frac 0.77, `--enable-dp-lm-head`
2. **sglang EP16+cap3100** — EP16 with mem-frac 0.77, `--max-running-requests 3100`, `--enable-dp-lm-head`
3. **sglang PP8TP2** — PP8 TP2, default mem-frac 0.80
4. **asyncmoe** — DisagMoE colocate EP16 DP16

### Profiles
- `sharegpt` — gating_gptoss120b_sharegpt_200.parquet
- `legal` — gating_legal_court_opinions_200.parquet

### Balanced Variants
- `original` — gating_profiles/<profile>.parquet
- `balanced` — gating_profiles/balanced_output/balanced_<profile>.parquet

### Fixed Parameters
- Model: gptoss-120b (gpt-oss-120b-bf16 for sglang, gptoss_120b for asyncmoe)
- Input/output length: 256–512 uniform
- 10,000 requests total, 2000 rps
- No advanced logging / recorder

## Benchmark Parameters

### sglang
```
python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --model lmsys/gpt-oss-120b-bf16 \
    --dataset-name random \
    --random-input-len 512 --random-output-len 512 --random-range-ratio 0.5 \
    --num-prompts 10000 --request-rate 2000 \
    --seed 1 --warmup-requests 1
```

### asyncmoe
```
curl -X POST http://localhost:6699/run_once \
    -H "Content-Type: application/json" \
    -d '{"rate": 2000, "time": 5, "distribution": "poisson",
         "min_input_len": 256, "max_input_len": 512,
         "min_output_len": 256, "max_output_len": 512}'
```

## Run Order (sglang first, then asyncmoe)

### Phase 1: sglang EP16 (4 runs)
| # | Profile | Balanced | Profile File |
|---|---------|----------|-------------|
| 1 | sharegpt | no | gating_gptoss120b_sharegpt_200.parquet |
| 2 | sharegpt | yes | balanced_output/balanced_gptoss120b_sharegpt_200.parquet |
| 3 | legal | no | gating_legal_court_opinions_200.parquet |
| 4 | legal | yes | balanced_output/balanced_legal_court_opinions_200.parquet |

Server flags: `--mem-fraction-static 0.77 --enable-dp-lm-head`

### Phase 2: sglang EP16+cap3100 (4 runs)
Same 4 profile combos.
Server flags: `--mem-fraction-static 0.77 --max-running-requests 3100 --enable-dp-lm-head`

### Phase 3: sglang PP8TP2 (4 runs)
Same 4 profile combos.
Server flags: default (mem-frac 0.80, no EP flags)

### Phase 4: asyncmoe (4 runs)
Same 4 profile combos.
Server: experiments/scripts/sphere-16/launch_server.sh with GATE_PROFILE_FILE modified per run.

## Retry Policy
- sglang: max 2 retries per config
- asyncmoe: max 3 retries per config

## Health Checks
- sglang: `curl -sf http://localhost:30000/health` (HTTP 200 = ready)
- asyncmoe: "Launching Flask Server" in server.log, then `curl -sf http://localhost:6699/`

## Progress Monitoring
- Check every 5 minutes
- asyncmoe: detokenizer throughput logs appear every ~second during normal operation; no logs for 5 min = hang → kill & retry
- sglang EP: benchmark < 5 min; PP8TP2: benchmark < 10 min

## Results Collection
After each run, write results to:
`/home/yizhuoliang/paper-repo/results/main-exp-mar23/<system>_<profile>_<balanced>.txt`

Metrics to capture:
- Output token throughput (tok/s)
- Mean ITL (ms)
- Median ITL (ms)
- P99 ITL (ms)
- Total benchmark duration (s)

Then git commit + push to origin.

## Log Locations
All server and benchmark logs: `/home/yizhuoliang/DisagMoE/experiments/amoe-064/logs/`

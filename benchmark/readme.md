# Benchmark usage

## Benchmark Scripts

### benchmark_serving.py

[TODO]

### benchmark_poisson.sh

Change the args in the script to run.

* `N_TIME`: the time of benchmark, in seconds.
* `step_attn`: PP degree for attention.
* `dp_size`: DP degree for attention.
* `step_exp`: PP degree for experts.
* `ep_size`: EP degree for experts.
* `RATE`: the rate of poisson distribution (in requests per second).
* `placement`: the placement method (`pipeline` or `colocate`).

### benchmark_e2e.sh

[TODO]

### launch_server.sh

A fast entrypoint for `server.py`. The args are almost the same as `benchmark_poisson.sh`.

### benchmark_instructions.sh

A simple curl command to send a benchmark request to the server. 

Request format:

```json
{
    "rate": <float>,
    "time": <int>,
    "distribution": <str>  // 'uniform' or 'poisson'
}
```

## Plotter Scripts

Each benchmark is associated with:
* `rate`
* `num_nodes`
* `dp_size`
* `ep_size`

### queueing_delay.py

An example:

```bash
python benchmark/plotter/queueing_delay.py --rate 30 --num-nodes 4 --dp-size 2 --ep-size 4
```

The results could be checked at `reports/throughput_benchmark/queueing_delay/`.

### sampler_step.py

An example:

```bash
python benchmark/plotter/sampler_step.py --rate 20 --gap-i 100 --num-nodes 4 --dp-size 2 --ep-size 4
```

The requests could be checked at `reports/throughput_benchmark/time/`

* `--gap-i`: [TODO]

### ttft.py

[TODO]

The requests could be checked at `reports/throughput_benchmark/ttft/`
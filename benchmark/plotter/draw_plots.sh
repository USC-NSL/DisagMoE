#!/usr/bin/bash

benchmark_dirs=(/home/shaoyuw/DisagMoE/reports/)

for benchmark_dir in "${benchmark_dirs[@]}"; do

working_dirs=($(ls -d $benchmark_dir/*))

for working_dir in "${working_dirs[@]}"; do
    if [[ ! -d "$working_dir" ]]; then
        continue
    fi
    python benchmark/plotter/output_req.py "$working_dir"
    python benchmark/plotter/sampler_step.py "$working_dir"
    python benchmark/plotter/queue_length.py "$working_dir"
done

done
#!/usr/bin/bash

benchmark_dir=$1
if [[ -z "$benchmark_dir" ]]; then
    echo "Usage: $0 <benchmark_dir>"
    exit 1
fi
if [[ ! -d "$benchmark_dir" ]]; then
    echo "Error: $benchmark_dir is not a directory"
    exit 1
fi

working_dirs=($(ls -d $benchmark_dir/*))

for working_dir in "${working_dirs[@]}"; do
    if [[ ! -d "$working_dir" ]]; then
        continue
    fi
    python benchmark/plotter/output_req.py "$working_dir"
    python benchmark/plotter/sampler_step.py --gap-t 5 "$working_dir"
    python benchmark/plotter/queue_length.py "$working_dir"

done
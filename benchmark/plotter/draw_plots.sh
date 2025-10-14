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
    bash benchmark/plotter/plot_trace.sh "$working_dir"
done
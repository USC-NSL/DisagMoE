#!/usr/bin/bash


working_dir=$1
if [[ -z "$working_dir" ]]; then
    echo "Usage: $0 <working_dir>"
    exit 1
fi
if [[ ! -d "$working_dir" ]]; then
    echo "Error: $working_dir is not a directory"
    exit 1
fi

python benchmark/plotter/output_req.py "$working_dir"
python benchmark/plotter/sampler_step.py --gap-t 5 "$working_dir"
python benchmark/plotter/queue_length.py "$working_dir"

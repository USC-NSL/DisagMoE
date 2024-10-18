#!/usr/bin/bash

ep="localhost"
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --rdzv_id=666 --rdzv_backend=c10d --rdzv_endpoint="$ep:29500" sender.py 

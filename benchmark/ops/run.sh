#!/bin/bash

set -x
which python
for e in 1; do
    for b in 40 100 200 400 800; do
        python grouped_experts_gemm.py -b $b -e $e
    done
done
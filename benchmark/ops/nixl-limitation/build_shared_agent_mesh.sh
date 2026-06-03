#!/bin/bash
set -euo pipefail
REPO=/home/yizhuoliang/DisagMoE
SRC=$REPO/benchmark/ops/nixl-limitation/shared_agent_mesh.cu
OUT=$REPO/benchmark/ops/nixl-limitation/shared_agent_mesh
NIXL_REPO=/tmp/nixl-repo
NIXL_BUILD=/tmp/nixl-repo/build-trace-wheelucx
CUDA_HOME=/usr/local/cuda
$CUDA_HOME/bin/nvcc -O2 -g -std=c++17 \
    -Xcompiler -pthread -arch=sm_89 \
    -I"$NIXL_REPO/src/bindings/rust" -I"$NIXL_REPO/src/api/cpp" -I"$CUDA_HOME/include" \
    -L"$NIXL_BUILD/src/bindings" -L"$NIXL_BUILD/src/core" -L"$CUDA_HOME/lib64" \
    -Xlinker -rpath,"$NIXL_BUILD/src/bindings" \
    -Xlinker -rpath,"$NIXL_BUILD/src/core" \
    -Xlinker -rpath,"$CUDA_HOME/lib64" \
    "$SRC" -o "$OUT" \
    -lnixl_capi -lnixl -lcudart
echo "built: $OUT"
ls -la "$OUT"

#!/bin/bash
# DisagMoE build and runtime environment for NCSA Delta
# Usage: source experiments/scripts/delta/env.sh

export PROJECT=/projects/bgro/spark36

conda activate amoe

# CUDA
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8

# NCCL 2.27 (compiled from source)
export NCCL_HOME=$PROJECT/nccl-2.27/build
export NCCL_INCLUDE_DIR=$NCCL_HOME/include
export NCCL_LIBRARY_DIR=$NCCL_HOME/lib

# ZeroMQ / GDRCopy
export ZMQ_HOME=$CONDA_PREFIX
export GDRCOPY_HOME=/usr

# UCX (HPC-X)
export UCX_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ucx
export C_INCLUDE_PATH=$UCX_DIR/include
export CPP_INCLUDE_PATH=$UCX_DIR/include

# Build paths
export LIBRARY_PATH=$CONDA_PREFIX/lib:$NCCL_HOME/lib:$UCX_DIR/lib:/usr/lib64:$LIBRARY_PATH

# Runtime paths (nvidia pip libs + torch + NCCL + UCX)
SITE=$CONDA_PREFIX/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cuda_cupti/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/nvtx/lib:$SITE/nvidia/nccl/lib:$SITE/torch/lib:$NCCL_HOME/lib:$UCX_DIR/lib:$CONDA_PREFIX/lib:/usr/lib64:$LD_LIBRARY_PATH

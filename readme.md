# Disag MoE

## Third party
See `.gitmodules`.

* cereal
* libzmq
* nvtx
* grouped_gemm

## Install dependencies

### Install dependencies

```bash
sudo apt-get install libzmq3-dev libcereal-dev
git submodule update --init
pip install -r requirements.txt
```

### Apply patch to vLLM

We hack and adopt the attention implementation of vLLM. A patch should be applied to the installed vllm 0.8.2 library.

```

cd path/to/vllm

git apply DisagMoE/patches/vllm_0.8.2.patch

```

### Build grouped_gemm

```bash
git submodule update --init
cd third_party/grouped_gemm
TORCH_CUDA_ARCH_LIST=8.0 GROUPED_GEMM_CUTLASS=1 pip install .
```

## Build C++ libraries

### Environments

To prepare for the compilation, make sure that these environment variables are properly set

- CUDA_HOME
- NCCL_HOME
- ZMQ_HOME (if libzmq is installed at user-level)

### Build

DisagMoE requires a c++ library built from `csrc` with setup.py following the command

```bash
make pip
```

It will leverage setup.py to build the shared library. The equivalent command is `pip install .`.

## Quick Start

```
export VLLM_FLASH_ATTN_VERSION=3

./benchmark/scripts/launch_server.sh
```

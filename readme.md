# Disag MoE

## Third party
See `.gitmodules`.

* cereal
* cppzmq/libzmq
* nvtx

## Install dependencies

### Install dependencies

```bash
sudo apt-get install libzmq3-dev libcereal-dev
git submodule update --init --recursive
pip install -r requirements.txt
```

### Apply patch to vLLM

We hack and adopt the attention implementation of vLLM. A patch should be applied to the installed vllm 0.8.2 library.

```

cd path/to/python-version/site-packages

git apply DisagMoE/patches/vllm_0.8.2.patch

```

### Build grouped_gemm

```bash
git submodule update --init
cd third_party/grouped_gemm
TORCH_CUDA_ARCH_LIST=8.0 GROUPED_GEMM_CUTLASS=1 pip install .
```


## Build C++ libraries

DisagMoE requires a c++ lib built from `csrc`. There are 2 ways to build it
- cmake
- setup.py

### With setup.py

```bash
make pip
```

It will leverage setup.py to build the shared library. The equivalent command is `pip install .`.

### With cmake

```bash
make cmake
```

This will build a shared library with cmake and install the library in the root directory of DisagMoE.

NOTE: The library built with cmake is under development and testing.

## Quick Start

```
ray start --head

./benchmark/scripts/launch_server.sh
```

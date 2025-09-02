# Disag MoE

## Third party
See `.gitmodules`.

* cereal
* cppzmq/libzmq
* nvtx


## Install

### Build grouped_gemm

CUDA 12.4 is required for grouped_gemm compilation.

```bash
git submodule update --init
cd third_party/grouped_gemm
TORCH_CUDA_ARCH_LIST=8.0 GROUPED_GEMM_CUTLASS=1 pip install .
```

### Build Disag-MoE

```bash
sudo apt-get install libzmq3-dev libcereal-dev
git submodule update --init
pip install -r requirements.txt
pip install .
```

## Tests

```bash
python tests/test_binding.py
```

## Build

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

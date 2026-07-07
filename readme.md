# StreamInfer

Barrier-free distributed Mixture-of-Experts serving system.

## Dependencies

### System

| Dependency | Version | Notes |
|---|---|---|
| CUDA Toolkit | 12.x | |
| GCC | 13+ | C++17 required |
| UCX | 1.15+ | For high-performance transport (`ucp`, `ucs`, `uct`), optional |
| GDRCopy | 2.x | GPUDirect RDMA (`libgdrapi`) |
| libzmq | 4.x | ZeroMQ C library; also needs cppzmq C++ headers (`zmq.hpp`) |

### Python (pip)

| Package | Version |
|---|---|
| Python | 3.12 |
| torch | 2.6.0 |
| vllm | 0.8.2 |
| pybind11 | latest |
| flask | latest |
| simpy | latest |
| pyarrow | latest |
| pandas | latest |
| openpyxl | latest |
| matplotlib | latest |

> **ABI Note**: Use the standard PyPI `torch` wheel (bundles CUDA 12.4, old C++ ABI).
> Do **not** install from `--index-url https://download.pytorch.org/whl/cu126` — that
> build uses CXX11 ABI, which causes `undefined symbol` errors when importing vLLM.

### Compiled from Source

| Dependency | Version | Why |
|---|---|---|
| NCCL | 2.27.x | Required version not available as pre-built package |

### Git Submodules (fetched automatically by `--recursive`)

- **CUTLASS** — grouped GEMM kernels, compiled into `disagmoe_c`
- **cereal** — C++ serialization (header-only)
- **NVTX** — NVIDIA profiling (header-only)
- **pybind11** — Python/C++ binding (header-only)

## Install

### 1. Python environment

```bash
conda create -n streaminfer python=3.12.8 -y
conda activate streaminfer
pip install torch==2.6.0 torchvision torchaudio
pip install vllm==0.8.2
```

### 2. NCCL 2.27

```bash
git clone --depth 1 --branch v2.27.7-1 https://github.com/NVIDIA/nccl.git nccl-2.27
cd nccl-2.27
make -j$(nproc) src.build \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80" \
    CUDA_HOME=$CUDA_HOME
cd ..
export NCCL_HOME=$(pwd)/nccl-2.27/build
```

### 3. ZeroMQ (C library + C++ headers)

With `apt`:

```bash
sudo apt-get install libzmq3-dev
```

Without root (e.g., HPC):

```bash
conda install -c conda-forge zeromq -y
wget -q -O $CONDA_PREFIX/include/zmq.hpp \
    https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp
wget -q -O $CONDA_PREFIX/include/zmq_addon.hpp \
    https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq_addon.hpp
```

### 4. Clone and install Python deps

```bash
git clone --recursive -b fp8_groupgemm https://github.com/USC-NSL/DisagMoE.git
cd DisagMoE
pip install -r requirements.txt
```

### 5. Build DisagMoE

```bash
export CUDA_HOME=/path/to/cuda
export NCCL_HOME=/path/to/nccl-2.27/build
export NCCL_INCLUDE_DIR=$NCCL_HOME/include
export NCCL_LIBRARY_DIR=$NCCL_HOME/lib
export ZMQ_HOME=$CONDA_PREFIX              # or wherever zmq.h and libzmq live
export GDRCOPY_HOME=/usr                   # or wherever gdrapi.h lives
export C_INCLUDE_PATH=/path/to/ucx/include
export CPP_INCLUDE_PATH=/path/to/ucx/include
export LIBRARY_PATH=$ZMQ_HOME/lib:$NCCL_HOME/lib:/path/to/ucx/lib:$LIBRARY_PATH

make pip
```

### 6. Verify

```bash
python -c "import disagmoe_c; print('OK')"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
```

## Tests

```bash
python tests/test_binding.py
```

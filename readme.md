# Disag MoE

## Third party
See `.gitmodules`.

* cereal
* libzmq
* nvtx
* cutlass (v3.2.0)
* ucx (not included as submodule)
* deep_gemm (not included as submodule)

## Install dependencies

### Install dependencies

```bash
sudo apt-get install libzmq3-dev libcereal-dev libucx-dev
git submodule update --init --recursive
pip install -r requirements.txt
```

NCCL:

```bash
sudo apt-get install -y libnccl2 libnccl-dev
```

GDRCopy:

```bash
sudo apt-get install -y flex bison
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make
sudo make prefix=/usr/local/gdrcopy install
sudo ldconfig
sudo bash ./insmod.sh
```

For `deep_gemm`, their pip install is currently broken. So please use their repo's `install.sh` on `v2.1.1` branch.

### Apply patch to vLLM

We hack and adopt the attention implementation of vLLM. A patch should be applied to the installed vllm 0.8.2 library.

```

cd path/to/python-version/site-packages

git apply DisagMoE/patches/vllm_0.8.2.patch

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
export VLLM_FLASH_ATTN_VERSION=3 # don't run this if GPU doesn't support

./benchmark/scripts/launch_server.sh
```

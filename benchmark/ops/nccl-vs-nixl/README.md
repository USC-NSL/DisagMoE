# NCCL vs NIXL P2P Microbenchmark

Cross-node P2P latency/throughput benchmark comparing NCCL (via a standalone NcclChannel C++ extension) and NIXL (UCX RDMA WRITE with GPU-Direct RDMA) over RoCE.

## Cluster Setup

**Nodes:** sgpu6 (10.0.0.1), sgpu7 (10.0.0.2), sgpu8 (10.0.0.3), sgpu9 (10.0.0.4)
**NIC:** ConnectX-6 200Gbps, interface `ens1f1np1`, HCA `mlx5_1`
**GPU:** NVIDIA L40S

## Prerequisites

### 1. MLNX_OFED (required for GPU-Direct RDMA)

The in-kernel `ib_core` does not export `ib_register_peer_memory_client`, which `nvidia_peermem` needs. MLNX_OFED replaces the RDMA stack with one that supports peer memory registration.

```bash
wget "https://content.mellanox.com/ofed/MLNX_OFED-24.10-1.1.4.0/MLNX_OFED_LINUX-24.10-1.1.4.0-ubuntu24.04-x86_64.tgz"
tar xzf MLNX_OFED_LINUX-24.10-1.1.4.0-ubuntu24.04-x86_64.tgz
cd MLNX_OFED_LINUX-24.10-1.1.4.0-ubuntu24.04-x86_64
sudo ./mlnxofedinstall --add-kernel-support --without-fw-update --force
sudo /etc/init.d/openibd restart
```

If `openibd restart` fails due to processes using RDMA devices, kill them first (`sudo pkill -f sglang` etc).

### 2. nvidia_peermem

After MLNX_OFED install, `nvidia_peermem` must be rebuilt against the new OFED headers via DKMS:

```bash
sudo dkms build nvidia/570.133.20 -k $(uname -r) --force
sudo dkms install nvidia/570.133.20 -k $(uname -r) --force
```

Then load the nvidia stack with `PeerMappingOverride` (required for GDR on some configs):

```bash
sudo rmmod nvidia_peermem nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"
sudo modprobe nvidia_uvm nvidia_drm nvidia_modeset nvidia_peermem
```

Persist across reboots:
```bash
echo 'options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"' | sudo tee /etc/modprobe.d/nvidia-peermem.conf
```

Verify:
```bash
lsmod | grep nvidia_peermem  # should show loaded
```

### 3. NIXL

```bash
pip install nixl
```

### 4. gdrcopy (if not already present)

Some nodes may be missing `libgdrapi.so`. Copy from a node that has it:
```bash
sudo cp -r /usr/local/gdrcopy /usr/local/gdrcopy
sudo ln -sf /usr/local/gdrcopy/lib/libgdrapi.so.2 /usr/local/lib/libgdrapi.so.2
sudo ldconfig
```

### 5. NCCL Extension (auto-built)

The `nccl_ext/` directory contains a standalone pybind11 module that JIT-compiles on first import via `torch.utils.cpp_extension.load()`. Requires:
- `CUDA_HOME` env var or `/usr/local/cuda-12.6` default
- `libnccl` on the linker path
- `torch` with C++ extension support

No manual build step needed.

## Verifying GPU-Direct RDMA

```python
import os
os.environ['UCX_LOG_LEVEL'] = 'diag'
os.environ['UCX_NET_DEVICES'] = 'mlx5_1:1'
from nixl._api import nixl_agent, nixl_agent_config
import torch; torch.cuda.set_device(0)
cfg = nixl_agent_config(backends=['UCX'])
a = nixl_agent('test', cfg)
buf = torch.randn(1024, dtype=torch.bfloat16, device='cuda:0')
a.register_memory(buf, backends=['UCX'])
```

If GDR is NOT working, you'll see:
```
GDAKI not supported, please load Nvidia peermem driver
mlx5_1: GPU-direct RDMA is not available
```

If working, no GDR-related diagnostic messages appear.

## Running

### 1:1 benchmark (sgpu6 ↔ sgpu7)

```bash
# Rsync to remote node first
rsync -av benchmark/ops/nccl-vs-nixl/ sgpu7:$(pwd)/benchmark/ops/nccl-vs-nixl/
bash benchmark/ops/nccl-vs-nixl/run.sh
```

### Fan-in benchmark (sgpu7,8,9 → sgpu6)

```bash
# Rsync to all sender nodes
for h in sgpu7 sgpu8 sgpu9; do
    rsync -av benchmark/ops/nccl-vs-nixl/ $h:$(pwd)/benchmark/ops/nccl-vs-nixl/
done
bash benchmark/ops/nccl-vs-nixl/run_fanin.sh
```

## Files

```
bench.py          - 1:1 sender/receiver benchmark
bench_fanin.py    - Fan-in N→1 synchronized benchmark (gloo barrier for NCCL, GO notification for NIXL)
run.sh            - Launcher for 1:1
run_fanin.sh      - Launcher for fan-in (3→1)
plot.py           - Line plots + box plot for 1:1 results
plot_fanin.py     - Aggregate plots for fan-in results
nccl_ext/         - Standalone NcclChannel pybind11 module (JIT-compiled)
```

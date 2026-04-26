# NCCL vs NIXL P2P Microbenchmark

Cross-node P2P latency/throughput benchmark comparing NCCL (via a standalone NcclChannel C++ extension) and NIXL (UCX RDMA WRITE with GPU-Direct RDMA) over RoCE.

## Cluster Setup

**Nodes:** sgpu6 (10.0.0.5), sgpu7 (10.0.0.6), sgpu8 (10.0.0.7), sgpu9 (10.0.0.8) on `ens1f1np1`
**NIC:** ConnectX-6 200Gbps, interface `ens1f1np1`, HCA `mlx5_1`
**GPU:** NVIDIA L40S

## Methodology

All ranks join a world-size gloo process group on the RoCE IP interface. The timed window is bracketed by `dist.barrier()` across every rank, so every sender's bench loop starts and ends within a few hundred microseconds of the others (no start-time drift). Earlier versions used a 100ms notification-poll loop for sync, which let per-sender windows under-overlap and produced aggregate throughput numbers above the physical link rate — that bug is fixed.

Cross-verification: the receiver reads the `mlx5_1` `port_rcv_data` IB counter before and after the timed window to get a ground-truth received-byte count. The sum of per-sender self-measured throughputs must agree with the receiver's HW-counter observed throughput within a few percent, and both must stay below 200 Gbps (25,000 MB/s). Large disagreement or super-link-rate numbers indicate a measurement bug. The plotter emits a cross-verification table and draws both the sender-aggregate and receiver-observed lines with a 200 Gbps reference line.

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

### 2. nvidia_peermem (CRITICAL for NIXL performance)

**Without `nvidia_peermem`, NIXL throughput drops from ~96% link rate to ~6%.**
The NIC cannot DMA directly from GPU memory, so every transfer bounces through
host RAM (GPU→CPU→NIC→CPU→GPU). This is the single most common cause of
unexpectedly low NIXL/UCX throughput. Always verify peermem is loaded before
benchmarking.

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

Verify on every node before running benchmarks:
```bash
lsmod | grep nvidia_peermem  # must show loaded
```

If `modprobe nvidia_peermem` fails with "Invalid argument", the full nvidia
stack must be reloaded with `PeerMappingOverride=1`. Kill all GPU processes
first, then run the `rmmod`/`modprobe` sequence above. If the modprobe.d
config is in place, a reboot also works.

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

### 5. UCCL P2P

UCCL is included as a git submodule under `uccl/`. Build and install the P2P
module on every node:

```bash
cd benchmark/ops/nccl-vs-nixl/uccl
git submodule update --init --recursive
bash build.sh cu12 p2p --install
```

Then copy the wheel to worker nodes:
```bash
WHEEL=$(ls uccl/wheelhouse-cu12/uccl-*.whl)
for h in sgpu7 sgpu8 sgpu9; do
    scp "$WHEEL" $h:/tmp/$(basename "$WHEEL")
    ssh $h "pip install /tmp/$(basename "$WHEEL")"
done
```

UCCL P2P uses its own RDMA transport (not NCCL). For RoCEv2 clusters, set
`UCCL_P2P_RDMA_GID_INDEX=3` (done automatically by `bench_fanin.py`).

### 6. NCCL Extension (auto-built)

The `nccl_ext/` directory contains a standalone pybind11 module that JIT-compiles on first import via `torch.utils.cpp_extension.load()`. Requires:
- `CUDA_HOME` env var or `/usr/local/cuda-12.6` default
- `libnccl` on the linker path
- `torch` with C++ extension support

No manual build step needed.

## Verifying GPU-Direct RDMA

**Run this on every node before benchmarking.** If GDR is off, NIXL will
silently fall back to host-memory staging at ~15× lower throughput.

```python
import os
os.environ['UCX_LOG_LEVEL'] = 'info'
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

Fix: `sudo modprobe nvidia_peermem` (see section 2 above).

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
bench.py          - 1:1 sender/receiver benchmark, gloo-barrier synced, HW-counter cross-verified
bench_fanin.py    - Fan-in N→1 benchmark (NCCL P2P, NCCL Gather, UCCL P2P, NIXL)
run.sh            - Launcher for 1:1
run_fanin.sh      - Launcher for fan-in
plot.py           - 1:1 plot: sender vs receiver HW-counter with 200Gbps reference
plot_fanin.py     - Fan-in plot: all backends with sender/receiver cross-verification
nccl_ext/         - Standalone NcclChannel pybind11 module (JIT-compiled)
uccl/             - UCCL submodule (build with `bash build.sh cu12 p2p --install`)
```

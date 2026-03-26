# DisagMoE Deployment on NCSA Delta (A100 GPUs)

## Prerequisites

- SLURM allocation on the `gpuA100x4` partition (4 A100-SXM4-40GB per node)
- Conda environment `amoe` with torch 2.6.0+cu124, vLLM 0.8.2, Ray 2.54
- DisagMoE source tree at `~/DisagMoE` on shared NFS (`/projects`)

## One-Time Setup

### 1. Allocate GPU nodes

```bash
sbatch delta_batch_hold.sh   # requests 4 nodes (16 GPUs)
squeue -u $USER              # find your JOBID and node list
```

### 2. Apply the vLLM patch

From a compute node (via `srun --pty bash`):

```bash
source ~/DisagMoE/experiments/scripts/delta/env.sh
VLLM_DIR=$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
cd $(dirname $VLLM_DIR)
git apply ~/DisagMoE/patches/vllm_0.8.2.patch
```

This patches vLLM to add `use_direct_call` support and disable the V1 engine.

### 3. Build the C++ extension

**Critical:** SLURM sets `TMPDIR=/tmp` on compute nodes. The build reads this variable for the hostfile directory. You must override it:

```bash
cd ~/DisagMoE
TMPDIR=/tmp/disagmoe/ make clean && TMPDIR=/tmp/disagmoe/ make pip
```

Verify the build:

```bash
strings disagmoe_c.cpython-312-x86_64-linux-gnu.so | grep /tmp
# Should show: /tmp/disagmoe/
# If it shows just /tmp, the build used the wrong TMPDIR — rebuild.
```

Since `~/DisagMoE` is on shared NFS, this build is visible from all nodes.

---

## Deploy on 4 GPUs (Single Node)

### 1. Get a shell on a compute node

```bash
srun --jobid=<JOBID> --nodelist=<NODE> --overlap --pty bash
```

### 2. Set up environment

```bash
source ~/DisagMoE/experiments/scripts/delta/env.sh
```

### 3. Start Ray (single node)

```bash
export RAY_TMPDIR=/tmp/ray
ray start --head --port=6379 --min-worker-port=30000 --max-worker-port=39999 --disable-usage-stats
ray status   # verify: 4 GPUs, 64 CPUs
```

### 4. Launch the server

```bash
cd ~/DisagMoE
bash experiments/scripts/delta/launch_server_local.sh
```

Configuration: `gptoss_120b`, 4 layers (reduced for testing), colocate placement, ZMQ transport, 4 dp/ep.

The server takes ~5 minutes to start (slow NFS imports). Wait for:

```
 * Running on http://0.0.0.0:6699
```

### 5. Run a benchmark

```bash
python3 -c "
import requests
r = requests.post('http://localhost:6699/run_once', json={
    'rate': 100, 'time': 2, 'distribution': 'poisson',
    'min_input_len': 32, 'max_input_len': 64,
    'min_output_len': 32, 'max_output_len': 64
}, timeout=300)
print(r.text)
"
```

### 6. Cleanup

```bash
# Ctrl+C the server, then:
ray stop
```

---

## Deploy on 16 GPUs (4 Nodes)

### 1. Get a shell on the head node

```bash
srun --jobid=<JOBID> --nodelist=<HEAD_NODE> --overlap --pty bash
source ~/DisagMoE/experiments/scripts/delta/env.sh
```

### 2. Start Ray head

```bash
export RAY_TMPDIR=/tmp/ray
ray start --head --port=6379 --min-worker-port=30000 --max-worker-port=39999 --disable-usage-stats
```

Ray binds to the management network IP by default. **Do not** use hsn0 (Slingshot) IPs for Ray — they don't support general TCP.

### 3. Start Ray workers on other nodes

From the **head node** shell, use `srun` to launch workers. The `sleep infinity` keeps the srun step alive so Ray daemons persist:

```bash
HEAD_IP=$(python3 -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0])")

for node in <WORKER_NODE_1> <WORKER_NODE_2> <WORKER_NODE_3>; do
  srun --jobid=<JOBID> --nodelist=$node --overlap bash -c \
    'eval "$(/path/to/miniconda3/bin/conda shell.bash hook)" && \
     conda activate amoe && \
     source ~/DisagMoE/experiments/scripts/delta/env.sh && \
     export RAY_TMPDIR=/tmp/ray && \
     ray stop 2>/dev/null && \
     ray start --address='$HEAD_IP':6379 --disable-usage-stats && \
     sleep infinity' &
done
```

Wait ~1 minute for workers to join, then verify:

```bash
ray status
# Should show: 256 CPU, 16 GPU, 4+ Active nodes
```

### 4. Launch the server

```bash
cd ~/DisagMoE
bash experiments/scripts/delta/launch_server_16gpu.sh
```

Configuration: `gptoss_120b`, 36 layers (full model), colocate placement, ZMQ transport, `--host-ifname hsn0` for data plane, 16 dp/ep.

Startup takes ~20 minutes:
- ~10 min: 16 workers importing Python packages over NFS (extreme I/O contention)
- ~5 min: engine initialization (NCCL channels, expert/attention executors, CUDA graphs)
- ~2 min: init_core and scheduler setup

Wait for:

```
 * Running on http://0.0.0.0:6699
```

### 5. Run a benchmark

```bash
python3 -c "
import requests
r = requests.post('http://localhost:6699/run_once', json={
    'rate': 100, 'time': 5, 'distribution': 'poisson',
    'min_input_len': 64, 'max_input_len': 128,
    'min_output_len': 64, 'max_output_len': 128
}, timeout=600)
print(r.text)
"
```

### 6. Cleanup

```bash
# Ctrl+C the server
ray stop
# Kill background srun workers:
kill %1 %2 %3 2>/dev/null
```

---

## Troubleshooting

### Workers crash with `get_ip_of_device` / `map::at` / `out_of_range`

The C++ extension was built with the wrong `TMPDIR`. Rebuild:

```bash
cd ~/DisagMoE
TMPDIR=/tmp/disagmoe/ make clean && TMPDIR=/tmp/disagmoe/ make pip
```

### OOM during engine initialization

Kill stale GPU processes from previous runs:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
kill -9 <PIDs>
```

Then reduce `-u` (GPU memory fraction) in the launch script, or reduce `MAX_BATCH_SIZE_ATTN` / `MAX_BATCH_SIZE_EXP`.

### Ray workers can't connect to head

- Use the **management network** IP for Ray, not the Slingshot/hsn0 IP.
- Launch workers via `srun` from the head node, not via `ssh` (intra-job SSH may not be enabled).
- Keep srun alive with `sleep infinity` after `ray start`.

### `FlashAttentionImpl` missing `use_direct_call`

The vLLM patch hasn't been applied:

```bash
cd <site-packages-dir>
git apply ~/DisagMoE/patches/vllm_0.8.2.patch
```

### Extremely slow startup

NFS I/O contention during Python imports. Each worker imports ~150 native extension modules. With 16 workers on 4 nodes, imports can take 10-15 minutes. This is normal for shared-filesystem HPC clusters.

---

## Delta-Specific Notes

- **GPU architecture**: A100 is sm_80, requires `--less-than-sm90` flag
- **Interconnect**: HPE Slingshot with CXI provider; NCCL uses `aws-ofi-nccl` via libfabric (not InfiniBand)
- **Ray network**: Must use management network (not hsn0) for Ray control plane; hsn0 is used for NCCL data plane via `--host-ifname hsn0`
- **Job scheduler**: SLURM — use `srun --overlap` to run commands on allocated nodes
- **TMPDIR override**: Required when building (`TMPDIR=/tmp/disagmoe/`) because SLURM overrides `TMPDIR=/tmp`

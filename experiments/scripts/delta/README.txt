Delta 4-Node A100 Setup & Installation Guide
==============================================

Cluster: 4 nodes × 4 A100-SXM4-40GB GPUs = 16 GPUs total
Partition: gpuA100x4
Account:   bgro-delta-gpu
Conda env: amoe (Python 3.12.8)
Project:   /projects/bgro/spark36/


1. Allocate Nodes
-----------------
Request a 4-node interactive job (or sbatch):

  salloc --account=bgro-delta-gpu --partition=gpuA100x4 \
    --nodes=4 --gpus-per-node=4 --time=18:00:00

Or use the hold script:

  sbatch /u/spark36/delta_batch_hold.sh


2. Install Miniconda
--------------------
All storage goes to /projects (500 GB quota). Home dir is only 100 GB.

  cd /projects/bgro/spark36/
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p /projects/bgro/spark36/miniconda3
  ln -sf /projects/bgro/spark36/miniconda3 ~/miniconda3

  # Accept TOS (required for conda >= 26.x)
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


3. Create Conda Environment
----------------------------

  conda create -n amoe python=3.12.8 -y
  conda activate amoe


4. Install PyTorch & vLLM
--------------------------

  pip install torch==2.6.0 torchvision torchaudio
  pip install vllm==0.8.2

  IMPORTANT: Use standard PyPI torch (cu124, old C++ ABI), NOT the cu126
  variant from --index-url https://download.pytorch.org/whl/cu126. The cu126
  build uses CXX11 ABI which causes "undefined symbol" errors with vLLM's
  pre-built wheel.


5. Compile NCCL 2.27.7 from Source
------------------------------------

  cd /projects/bgro/spark36/
  git clone --depth 1 --branch v2.27.7-1 https://github.com/NVIDIA/nccl.git nccl-2.27
  cd nccl-2.27
  make -j$(nproc) src.build \
      NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80" \
      CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
  cd ..


6. Install ZeroMQ
------------------
No apt-get on Delta. Use conda for the C library, download C++ headers manually.

  conda install -c conda-forge zeromq -y
  wget -q -O $CONDA_PREFIX/include/zmq.hpp \
      https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp
  wget -q -O $CONDA_PREFIX/include/zmq_addon.hpp \
      https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq_addon.hpp


7. Install Python Dependencies
-------------------------------

  pip install pybind11 pandas openpyxl matplotlib flask simpy pyarrow


8. Clone DisagMoE (fp8_groupgemm branch)
------------------------------------------

  cd /projects/bgro/spark36/
  git clone --recursive -b fp8_groupgemm https://github.com/USC-NSL/DisagMoE.git
  cd DisagMoE


9. Build DisagMoE (make pip)
-----------------------------
Must run on a GPU node via srun. Set all env vars first:

  srun --jobid=<JOBID> --nodelist=<NODE> bash

Then inside the node:

  eval "$(/projects/bgro/spark36/miniconda3/bin/conda shell.bash hook)"
  conda activate amoe

  export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
  export NCCL_HOME=/projects/bgro/spark36/nccl-2.27/build
  export NCCL_INCLUDE_DIR=$NCCL_HOME/include
  export NCCL_LIBRARY_DIR=$NCCL_HOME/lib
  export ZMQ_HOME=$CONDA_PREFIX
  export GDRCOPY_HOME=/usr

  # UCX from HPC-X
  UCX_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ucx
  export C_INCLUDE_PATH=$UCX_DIR/include
  export CPP_INCLUDE_PATH=$UCX_DIR/include
  export LIBRARY_PATH=$CONDA_PREFIX/lib:$NCCL_HOME/lib:$UCX_DIR/lib:/usr/lib64:$LIBRARY_PATH
  export LD_LIBRARY_PATH=$NCCL_HOME/lib:$CONDA_PREFIX/lib:$UCX_DIR/lib:/usr/lib64:$LD_LIBRARY_PATH

  cd /projects/bgro/spark36/DisagMoE
  make pip

To recompile only (after code changes):

  cd /projects/bgro/spark36/DisagMoE && make clean && make pip


10. Verify Installation
------------------------

  python -c "import disagmoe_c; print('disagmoe_c OK')"
  python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.is_available()}')"
  python -c "import vllm; print(f'vllm={vllm.__version__}')"

Expected output:

  disagmoe_c OK
  torch=2.6.0+cu124, cuda=12.4, gpu=True
  vllm=0.8.2


Runtime LD_LIBRARY_PATH
------------------------
Must be set before running DisagMoE in every session:

  SITE=$(python -c "import site; print(site.getsitepackages()[0])")
  UCX_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ucx
  export LD_LIBRARY_PATH=\
  $SITE/nvidia/nvjitlink/lib:\
  $SITE/nvidia/cusparse/lib:\
  $SITE/nvidia/cublas/lib:\
  $SITE/nvidia/cuda_runtime/lib:\
  $SITE/nvidia/cudnn/lib:\
  $SITE/nvidia/cufft/lib:\
  $SITE/nvidia/curand/lib:\
  $SITE/nvidia/cusolver/lib:\
  $SITE/nvidia/cuda_cupti/lib:\
  $SITE/nvidia/cuda_nvrtc/lib:\
  $SITE/nvidia/nvtx/lib:\
  $SITE/nvidia/nccl/lib:\
  $SITE/torch/lib:\
  $NCCL_HOME/lib:\
  $UCX_DIR/lib:\
  $CONDA_PREFIX/lib:\
  $LD_LIBRARY_PATH


Notes
-----
- Delta uses Kerberos + Duo MFA for SSH. ControlMaster in ~/.ssh/config
  keeps the session alive for 24h after initial login.
- CUDA 12.8 toolkit is loaded by default (module cudatoolkit/25.3_12.8).
  PyTorch bundles its own CUDA 12.4 runtime; the 12.8 driver is compatible.
- /projects has 500 GB quota; /work/hdd has 1 TB (expandable to 100 TB).
  Use /projects for conda and source, /work/hdd for large datasets.
- GDRCopy 2.5 and UCX 1.15.0 are pre-installed on compute nodes.
- The fp8_groupgemm branch compiles grouped GEMM directly into disagmoe_c
  via CUTLASS. No separate grouped_gemm pip install is needed.

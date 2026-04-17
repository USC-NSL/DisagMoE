import os
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.6")

_mod = load(
    name="bench_nccl",
    sources=[os.path.join(_dir, "bench_nccl.cpp")],
    extra_include_paths=[os.path.join(_cuda_home, "targets/x86_64-linux/include")],
    extra_ldflags=["-lnccl", f"-L{_cuda_home}/lib64", "-lcudart"],
    verbose=False,
)

NcclChannel = _mod.NcclChannel
get_nccl_unique_id_bytes = _mod.get_nccl_unique_id_bytes

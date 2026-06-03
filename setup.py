from setuptools import setup, Extension, find_packages
import pybind11
import subprocess
import torch
from torch.utils import cpp_extension

subprocess.run(["g++", "--version"])

from pybind11.setup_helpers import build_ext, Pybind11Extension

import os

CSRC_DIR = os.path.abspath("csrc")
THIRD_PARTY_DIR = os.path.abspath("third_party")

ENABLE_NIXL = os.environ.get("ENABLE_NIXL", "0") == "1"

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
CUDA_INCLUDE_DIR = os.environ.get("CUDA_INCLUDE_DIR", os.path.join(CUDA_HOME, "include"))
CUDA_LIBRARY_DIR = os.environ.get("CUDA_LIBRARY_DIR", os.path.join(CUDA_HOME, "lib"))
CUDA_LIB64_DIR = os.environ.get("CUDA_LIBRARY_DIR", os.path.join(CUDA_HOME, "lib64"))

NCCL_HOME = os.environ.get("NCCL_HOME", "/usr/local/nccl2")
NCCL_INCLUDE_DIR = os.environ.get("NCCL_INCLUDE_DIR", os.path.join(NCCL_HOME, "include"))
NCCL_LIBRARY_DIR = os.environ.get("NCCL_LIBRARY_DIR", os.path.join(NCCL_HOME, "lib"))

TORCH_HOME = torch.__path__[0]
TORCH_LIB_DIR = f"{TORCH_HOME}/lib"
TORCH_INCLUDES = [f"{TORCH_HOME}/include/torch/csrc/api/include", f"{TORCH_HOME}/include"]

C_INCLUDE_PATH = os.environ.get("C_INCLUDE_PATH", "")
CPP_INCLUDE_PATH = os.environ.get("CPP_INCLUDE_PATH", "")
LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", "")

ZMQ_HOME = os.environ.get("ZMQ_HOME", "")
ZMQ_INCLUDE_PATH = os.path.join(ZMQ_HOME, "include")
ZMQ_LIBRARY_PATH = os.path.join(ZMQ_HOME, "lib")
TMPDIR=os.environ.get("TMPDIR", "/tmp/disagmoe/")

GDRCOPY_HOME = os.environ.get("GDRCOPY_HOME", "/usr/local/gdrcopy")
GDRCOPY_INCLUDE_DIR = os.path.join(GDRCOPY_HOME, "include")
GDRCOPY_LIBRARY_DIR = os.path.join(GDRCOPY_HOME, "lib")
KERNEL_USE_GDRCOPY = os.environ.get("KERNEL_USE_GDRCOPY", "1")
D_ENABLE_HANG_DEBUGGER = os.environ.get("D_ENABLE_HANG_DEBUGGER", "0")

def _detect_nixl_lib_dir():
    try:
        import sysconfig
        import glob
        site = sysconfig.get_paths().get("purelib")
        if site:
            for pat in (".nixl_cu12.mesonpy.libs", "nixl_cu12.libs", "nixl.libs"):
                hits = glob.glob(os.path.join(site, pat))
                if hits:
                    return hits[0]
    except Exception:
        pass
    return None

NIXL_HOME = os.environ.get("NIXL_HOME", "")
NIXL_INCLUDE_DIR = os.environ.get(
    "NIXL_INCLUDE_DIR",
    os.path.join(NIXL_HOME, "src", "api", "cpp") if NIXL_HOME else "",
)
NIXL_CAPI_INCLUDE_DIR = os.environ.get(
    "NIXL_CAPI_INCLUDE_DIR",
    os.path.join(NIXL_HOME, "src", "bindings", "rust") if NIXL_HOME else "",
)
NIXL_LIBRARY_DIR = os.environ.get("NIXL_LIBRARY_DIR", "") or (_detect_nixl_lib_dir() or "")

extra_includes = []
extra_libdirs = []
extra_libs = []
extra_macros = []
extra_link_args = []

if ENABLE_NIXL:
    missing = [name for name, val in [
        ("NIXL_INCLUDE_DIR", NIXL_INCLUDE_DIR),
        ("NIXL_CAPI_INCLUDE_DIR", NIXL_CAPI_INCLUDE_DIR),
        ("NIXL_LIBRARY_DIR", NIXL_LIBRARY_DIR),
    ] if not val]
    if missing:
        raise RuntimeError(
            "ENABLE_NIXL=1 but the following env vars are unset and could not "
            "be auto-detected: " + ", ".join(missing) + ". Set NIXL_HOME to "
            "point at a NIXL source checkout (defines NIXL_INCLUDE_DIR and "
            "NIXL_CAPI_INCLUDE_DIR), or set each path explicitly. "
            "NIXL_LIBRARY_DIR is auto-detected from the pip-installed "
            "nixl_cu12 package's .libs directory if available."
        )
    extra_includes.append(NIXL_INCLUDE_DIR)
    extra_includes.append(NIXL_CAPI_INCLUDE_DIR)
    extra_libdirs.append(NIXL_LIBRARY_DIR)
    extra_libs.append("nixl")
    extra_libs.append("nixl_capi")
    extra_macros.append(("USE_NIXL", "1"))
    extra_link_args.append(f"-Wl,-rpath,{NIXL_LIBRARY_DIR}")
else:
    extra_macros.append(("USE_NIXL", "0"))

def find_all_c_targets(path):
    res = []
    for root, dirs, files in os.walk(path):
        if "build" in root:
            continue
        for file_name in files:
            if file_name.endswith(".cpp") or file_name.endswith(".cu"):
                res.append(os.path.join(root, file_name))
    print(res)
    return res

THIRD_PARTY_INCLUDES = [
    f"{THIRD_PARTY_DIR}/cereal/include",
    f"{THIRD_PARTY_DIR}/NVTX/c/include",
    f"{THIRD_PARTY_DIR}/pybind11/include",
    f"{THIRD_PARTY_DIR}/cutlass/include",
]

ext_modules = [
    cpp_extension.CppExtension(
        'disagmoe_c',
        find_all_c_targets(CSRC_DIR),
        include_dirs=[d for d in [
            pybind11.get_include(),
            os.path.join(CSRC_DIR, "include"),
            CUDA_INCLUDE_DIR,
            NCCL_INCLUDE_DIR,
            ZMQ_INCLUDE_PATH,
            *THIRD_PARTY_INCLUDES,
            *TORCH_INCLUDES,
            C_INCLUDE_PATH,
            CPP_INCLUDE_PATH,
            GDRCOPY_INCLUDE_DIR,
            *extra_includes,
        ] if d],
        library_dirs=[d for d in [
            CUDA_LIBRARY_DIR,
            CUDA_LIB64_DIR,
            TORCH_LIB_DIR,
            NCCL_LIBRARY_DIR,
            ZMQ_LIBRARY_PATH,
            LD_LIBRARY_PATH,
            GDRCOPY_LIBRARY_DIR,
            "/usr/local/lib",
            "/usr/lib",
            *extra_libdirs,
        ] if d],
        libraries=["cudart", "nccl", "zmq", "ucp", "ucs", "uct", "torch", "c10", "torch_cpu", "gdrapi", *extra_libs],
        extra_compile_args=["-lstdc++", "-O2", "-w", "-std=c++17"],
        extra_link_args=extra_link_args,
        define_macros=[
            ("D_ENABLE_RAY", "1"),
            ("D_ENABLE_NVTX", "1"),
            ("D_GROUP_NCCL_RECV", "0"),
            ("TEMP_DIR", f'"{TMPDIR}"'),
            ("KERNEL_USE_GDRCOPY", KERNEL_USE_GDRCOPY),
            ("D_ENABLE_HANG_DEBUGGER", D_ENABLE_HANG_DEBUGGER),
            *extra_macros,
        ],
        language='c++',
    ),
]

setup(
    name='disagmoe',
    version='0.3.1',
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    ext_modules=ext_modules,
    packages=find_packages(".")
)

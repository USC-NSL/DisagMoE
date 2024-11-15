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

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
NCCL_HOME = os.environ.get("NCCL_HOME", "/usr/local/nccl2")

CUDA_INCLUDE_DIR = os.environ.get("CUDA_INCLUDE_DIR", os.path.join(CUDA_HOME, "include"))
CUDA_LIBRARY_DIR = os.environ.get("CUDA_LIBRARY_DIR", os.path.join(CUDA_HOME, "lib"))
CUDA_LIB64_DIR = os.environ.get("CUDA_LIBRARY_DIR", os.path.join(CUDA_HOME, "lib64"))
NCCL_LIB_DIR = os.environ.get("NCCL_LIBRARY_DIR", os.path.join(NCCL_HOME, "lib"))

TORCH_HOME = torch.__path__[0]
TORCH_LIB_DIR = f"{TORCH_HOME}/lib"
TORCH_INCLUDES = [f"{TORCH_HOME}/include/torch/csrc/api/include", f"{TORCH_HOME}/include"]

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

ext_modules = [
    cpp_extension.CppExtension(
        'disagmoe_c',
        find_all_c_targets(CSRC_DIR),
        include_dirs=[
            pybind11.get_include(),
            os.path.join(CSRC_DIR, "includes"),
            CUDA_INCLUDE_DIR,
            f"{THIRD_PARTY_DIR}/zmq/include",  # NOTE(hogura|20240927): if already installed in apt, this could be skipped
            f"{THIRD_PARTY_DIR}/cereal/include",
            f"{THIRD_PARTY_DIR}/NVTX/c/include",
            f"{NCCL_HOME}/include",
            # *TORCH_INCLUDES,
       ],
        library_dirs=[
            CUDA_LIBRARY_DIR,
            CUDA_LIB64_DIR,
            NCCL_LIB_DIR,
            TORCH_LIB_DIR,
        ], 
        libraries=["cudart", "nccl", "zmq", "torch", "c10", "torch_cpu"],
        extra_compile_args=["-lstdc++", "-O0", "-g"],
        define_macros=[
            ("D_ENABLE_RAY", "1"),
            ("D_ENABLE_NVTX", "1"),
        ],
        language='c++',
    ),
]

setup(
    name='disagmoe',
    version='0.2',
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    ext_modules=ext_modules,
    packages=find_packages(".")
)

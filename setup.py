from setuptools import setup, Extension, find_packages
import pybind11
import subprocess

subprocess.run(["g++", "--version"])

from pybind11.setup_helpers import build_ext, Pybind11Extension

import os

CSRC_DIR = "disagmoe/csrc"

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
CUDA_INCLUDE_DIR = os.environ.get("CUDA_INCLUDE_DIR", os.path.join(CUDA_HOME, "include"))
CUDA_LIBRARY_DIR = os.environ.get("CUDA_LIBRARY_DIR", os.path.join(CUDA_HOME, "lib"))

def find_all_c_targets(path):
    res = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if len(file_name) > 4 and file_name[-4:] == ".cpp":
                res.append(os.path.join(root, file_name))
    print(res)
    return res

ext_modules = [
    Pybind11Extension(
        'disagmoe_c',
        find_all_c_targets(CSRC_DIR),
        include_dirs=[
            pybind11.get_include(),
            os.path.join(CSRC_DIR, "includes"),
            CUDA_INCLUDE_DIR,
            "third_party/zmq/include",  # NOTE(hogura|20240927): if already installed in apt, this could be skipped
            "third_party/cereal/include",
        ],
        library_dirs=[
            CUDA_LIBRARY_DIR,
        ],
        libraries=["cudart", "nccl", "zmq"],
        extra_compile_args=["-lstdc++", "-O0", "-g"],
        define_macros=[
            ("D_ENABLE_RAY", "1"),
        ],
        language='c++',
    ),
]

setup(
    name='disagmoe',
    version='0.2',
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    packages=find_packages(".")
)

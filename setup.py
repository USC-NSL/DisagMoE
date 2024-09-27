from setuptools import setup, Extension
import pybind11

from pybind11.setup_helpers import build_ext, Pybind11Extension

import os

CSRC_DIR = "disagmoe/csrc"

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
CUDA_INCLUDE_DIR = os.environ.get("CUDA_INCLUDE_DIR", os.path.join(CUDA_HOME, "include"))
CUDA_LIBRARY_DIR = os.environ.get("CUDA_INCLUDE_DIR", os.path.join(CUDA_HOME, "lib"))

NCCL_INCLUDE_DIR = os.environ.get("NCCL_INCLUDE_DIR", "/usr/include")

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
            CUDA_INCLUDE_DIR,
            NCCL_INCLUDE_DIR,
            os.path.join(CSRC_DIR, "includes"),
        ],
        library_dirs=[
            CUDA_LIBRARY_DIR,
        ],
        language='c++',
    ),
]

setup(
    name='disagmoe',
    version='0.2',
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)

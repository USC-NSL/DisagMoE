#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.6.3 2>/dev/null || true

if ! command -v cmake &> /dev/null; then
    module load cmake/3.29.4 2>/dev/null || module load cmake 2>/dev/null || true
fi

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please load CUDA module."
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found. Please load a CMake module."
    exit 1
fi

echo "Using CUDA: $(which nvcc)"
nvcc --version

# Default to SM80 (A100). For Hopper (H100/H200), prefer 90a.
# Override with:
#   CUDA_ARCH=80  ./build.sh
#   CUDA_ARCH=90a ./build.sh
CUDA_ARCH="${CUDA_ARCH:-80}"
echo "Target architecture: ${CUDA_ARCH}"

mkdir -p build
cd build

echo ""
echo "Configuring (FetchContent downloads CUTLASS v3.2.0 on first build)..."

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

echo ""
echo "Building..."
make -j"$(nproc)"

echo ""
echo "Build successful!"
echo "Executable: $SCRIPT_DIR/build/moe_gemm_profiler"
echo ""
echo "To run the profiler:"
echo "  cd $SCRIPT_DIR/build && ./moe_gemm_profiler"

#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Ensure CUDA is installed/loaded."
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found. Install CMake (>= 3.18)."
    exit 1
fi

echo "Using CUDA: $(which nvcc)"
nvcc --version
echo "Using CMake: $(which cmake)"
cmake --version

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

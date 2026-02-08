cmake_minimum_required(VERSION 3.18)
project(moe_gemm_profiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to SM80 (A100), can be overridden via -DCMAKE_CUDA_ARCHITECTURES=90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80")
endif()

# ==============================================================================
# CUTLASS Dependency via FetchContent
# ==============================================================================
include(FetchContent)

FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        v3.2.0
    GIT_SHALLOW    TRUE
)

# Don't build CUTLASS tests/examples, just use headers
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(cutlass)

# ==============================================================================
# Include directories
# ==============================================================================
include_directories(
    ${cutlass_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/tools/util/include
    ${CMAKE_SOURCE_DIR}
)

# ==============================================================================
# Compiler flags
# ==============================================================================
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1")

# ==============================================================================
# Executable
# ==============================================================================
add_executable(moe_gemm_profiler
    profiler.cu
)

target_link_libraries(moe_gemm_profiler
    cuda
    cudart
)

set_target_properties(moe_gemm_profiler PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

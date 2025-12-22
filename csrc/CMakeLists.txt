cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER nvcc)

project(disagmoe_c LANGUAGES CUDA CXX)

set(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(THIRD_PARTY_DIR "${PROJECT_DIR}/third_party")
set(CMAKE_INSTALL_PREFIX ${PROJECT_DIR})
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(${THIRD_PARTY_DIR}/pybind11 pybind11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++ -std=c++17 -O0 -g -D D_ENABLE_RAY=1 -D D_ENABLE_NVTX=1")
message(CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS})

set(NCCL_HOME /usr/local/nccl2)
message(NCCL_HOME=${NCCL_HOME})

set(CUDA_HOME /usr/local/cuda)
message(CUDA_HOME=${CUDA_HOME})

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
## load in pkg-config support
find_package(PkgConfig)
## use pkg-config to get hints for 0mq locations
pkg_check_modules(PC_ZeroMQ QUIET zmq)
pkg_check_modules(PC_UCX QUIET ucx)

## use the hint from above to find where 'zmq.hpp' is located
find_path(ZeroMQ_INCLUDE_DIR
        NAMES zmq.hpp zmq.h
        PATHS ${PC_ZeroMQ_INCLUDE_DIRS}
        )

## use the hint from above to find the location of libzmq
find_library(ZeroMQ_LIBRARY
        NAMES zmq
        PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
        )

find_library(NCCL_LIBRARY
        NAMES nccl
        PATHS ${NCCL_HOME}/lib
)

message(ZMQ_INCLUDE=${ZeroMQ_INCLUDE_DIR})
message(ZMQ_LIBRARY=${ZeroMQ_LIBRARY})
message(UCX_INCLUDE=${UCX_INCLUDE_DIR})
message(UCX_UCP_LIBRARY=${UCX_UCP_LIBRARY})
message(UCX_UCS_LIBRARY=${UCX_UCS_LIBRARY})
message(UCX_UCT_LIBRARY=${UCX_UCT_LIBRARY})

if (NOT UCX_INCLUDE_DIR OR NOT UCX_UCP_LIBRARY OR NOT UCX_UCS_LIBRARY OR NOT UCX_UCT_LIBRARY)
    message(FATAL_ERROR "UCX development package not found. Please install libucx and ensure pkg-config is configured.")
endif()

file(
    GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS
    "engine/*.cpp"
    "muhelper/*.cpp"
    "debugger/*.cpp"
    "include/transport_factory.h"
    "bindings.cpp"
)

# message(sources=${CPP_SOURCES})

pybind11_add_module(disagmoe_c ${CPP_SOURCES})
target_link_options(disagmoe_c PRIVATE -rdynamic)

target_include_directories(disagmoe_c PUBLIC includes)
target_include_directories(disagmoe_c PUBLIC ${THIRD_PARTY_DIR}/cereal/include)
target_include_directories(disagmoe_c PUBLIC ${THIRD_PARTY_DIR}/NVTX/c/include)
target_include_directories(disagmoe_c PUBLIC ${THIRD_PARTY_DIR}/pybind11/include)
target_include_directories(disagmoe_c PUBLIC ${NCCL_HOME}/include)
target_include_directories(disagmoe_c PUBLIC ${CUDA_HOME}/include)
target_include_directories(disagmoe_c PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(disagmoe_c PUBLIC ${ZeroMQ_INCLUDE_DIR})
target_include_directories(disagmoe_c PUBLIC ${UCX_INCLUDE_DIR})

target_link_libraries(disagmoe_c PUBLIC CUDA::cudart)
target_link_libraries(disagmoe_c PUBLIC ${NCCL_LIBRARY})
target_link_libraries(disagmoe_c PUBLIC ${ZeroMQ_LIBRARY})
target_link_libraries(disagmoe_c PUBLIC ${UCX_UCP_LIBRARY} ${UCX_UCS_LIBRARY} ${UCX_UCT_LIBRARY})
target_link_libraries(disagmoe_c PUBLIC pybind11::module)

install(TARGETS disagmoe_c DESTINATION ${PROJECT_DIR})

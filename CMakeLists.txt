cmake_minimum_required(VERSION 3.20)
project(MonteCarloGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Target the common data center and desktop architectures. Ampere (86) is the
# default. Override with -DCMAKE_CUDA_ARCHITECTURES=90 for Hopper etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# Core engine library shared by the CLI, tests, and benchmarks.
add_library(mc_engine STATIC
    src/black_scholes.cpp
    src/pricer.cpp
    src/kernels/european.cu
    src/kernels/asian.cu
    src/kernels/barrier.cu
)
set_target_properties(mc_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
# cuRAND ships with the CUDA toolkit.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(mc_engine PUBLIC CUDA::curand)

# Command line pricer.
add_executable(montecarlo_gpu src/main.cpp)
target_link_libraries(montecarlo_gpu PRIVATE mc_engine)

# Optional test and benchmark trees. Off by default so the container build
# stays lean, on in CI and local development.
option(MC_BUILD_TESTS "Build the Google Test suite" OFF)
option(MC_BUILD_BENCH "Build the Google Benchmark microbenchmarks" OFF)

if(MC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(MC_BUILD_BENCH)
    add_subdirectory(bench)
endif()

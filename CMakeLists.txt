cmake_minimum_required(VERSION 3.20)

project(
    cuda_nn
    LANGUAGES CXX CUDA
)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Use CUDA C++17 as well
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Nice default while developing
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(cuda_nn
    src/main.cpp
    src/matrix.cpp
    src/linear.cpp
    src/relu.cpp

    src/cuda/device_matrix.cu
    src/cuda/ops.cu

    src/cuda/kernel/relu.cu
    src/cuda/kernel/matmul.cu
    
)

# Add header directory
target_include_directories(cuda_nn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# # Let CMake know this target contains CUDA code
set_target_properties(cuda_nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
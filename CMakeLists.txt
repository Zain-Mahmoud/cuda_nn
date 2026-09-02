cmake_minimum_required(VERSION 3.20)

project(
    cuda_nn
    LANGUAGES CXX CUDA
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(cuda_nn
    src/main.cpp
    src/matrix.cpp
    src/linear.cpp
    src/relu.cpp

    src/cuda/device_matrix.cu
    src/cuda/ops.cu

    src/cuda/kernels/relu.cu
    src/cuda/kernels/swish.cu
    src/cuda/kernels/gelu.cu
    src/cuda/kernels/softmax.cu
    src/cuda/kernels/tanh.cu
    src/cuda/kernels/matmul.cu
    
)


target_include_directories(cuda_nn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


set_target_properties(cuda_nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
#include <relu.cuh>
#include <matmul.cuh>
#include <device_matrix.hpp>
#include <iostream>
#include <cmath>

using namespace std;

void relu_gpu(DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    relu_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA RELU Kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }

}

void matmul_gpu(DeviceMatrix &out, DeviceMatrix& left, DeviceMatrix& right){
    int total_threads = left.rows * right.cols;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    int M = left.rows;
    int K = left.cols;
    int N = right.cols;

    naive_matmul_kernel<<<blocks_per_grid, threads_per_block>>>(out.data, left.data, right.data, M, K, N);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA naive matmul kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }

}

void matmul_tiled_gpu(DeviceMatrix &out, DeviceMatrix& left, DeviceMatrix& right){

    int M = left.rows;
    int K = left.cols;
    int N = right.cols;

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(ceil((float)N / TILE_SIZE), ceil((float)M /TILE_SIZE));

    tiled_matmul_kernel<<<gridSize, blockSize>>>(out.data, left.data, right.data, M, K, N);
    
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA tiled matmul Kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }

}


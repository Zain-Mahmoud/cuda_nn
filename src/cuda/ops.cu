#include <relu.cuh>
#include <matmul.cuh>
#include <softmax.cuh>
#include <tanh.cuh>
#include <gelu.cuh>
#include <swish.cuh>
#include <device_matrix.hpp>
#include <iostream>
#include <cmath>
#include <cassert>


using namespace std;

int next_power_of_two(int N) {
    int power = 1;

    while (power < N) {
        power *= 2;
    }

    return power;
}

void relu_gpu(DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    relu_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA RELU kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}

void relu_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    relu_grad_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA RELU grad kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}


void matmul_gpu(DeviceMatrix &out, DeviceMatrix& left, DeviceMatrix& right){
    assert(left.cols == right.rows);
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
    assert(left.cols == right.rows);
    int M = left.rows;
    int K = left.cols;
    int N = right.cols;

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(ceil((float)N / TILE_SIZE), ceil((float)M /TILE_SIZE));

    tiled_matmul_kernel<<<gridSize, blockSize>>>(out.data, left.data, right.data, M, K, N);
    
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA tiled matmul kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }

}

void softmax_gpu(DeviceMatrix &X, DeviceMatrix &Y){
    assert(X.cols == 1);
    assert(Y.cols == 1);

    int size = X.rows;

    if (size > BLOCKSIZE){
        return;
    }

    float *d_max;
    float *d_sum;

    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    int threads = next_power_of_two(size);

    max_kernel<<<1, threads>>>(d_max, X.data, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        cerr << "CUDA softmax maximum kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }

    sum_kernel<<<1, threads>>>(d_sum, X.data, size, d_max);

    err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA softmax sum kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }

    softmax_kernel<<<1, threads>>>(Y.data, X.data, size, d_max, d_sum);
    err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA softmax calculation kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}

void softmax_grad_gpu(DeviceMatrix &X, DeviceMatrix &Y){
    assert(X.cols == 1);

    int size = X.rows;

    if (size > BLOCKSIZE){
        return;
    }
    int threads = 256;
    int total = size * size;
    int blocks = (total + threads - 1) / threads;

    softmax_grad_kernel<<<blocks, threads>>>(Y.data, X.data, size);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA softmax grad calculation kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}


void tanh_gpu(DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    tanh_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA tanh kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}

void tanh_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    tanh_grad_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA tanh grad kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}

void leaky_relu_gpu(DeviceMatrix& X, DeviceMatrix& Y, float alpha){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    leaky_relu_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size, alpha);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA leaky RELU kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}

void leaky_relu_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y, float alpha){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    leaky_relu_grad_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size, alpha);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA leaky RELU grad kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}

void gelu_gpu(DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    gelu_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA GELU kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}

void gelu_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    gelu_grad_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA GELU grad kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}

void siwsh_gpu(DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    swish_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA swish kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}

void siwsh_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    swish_grad_kernel<<<blocks_per_grid, threads_per_block>>>(Y.data, X.data, matrix_size);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        cerr << "CUDA swish grad kernel launch failed: "
        << cudaGetErrorString(err) << '\n';
    }
}


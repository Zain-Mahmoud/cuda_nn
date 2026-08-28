#include <relu.cuh>
#include <matmul.cuh>
#include <device_matrix.hpp>

void relu_gpu(const DeviceMatrix& X, DeviceMatrix& Y){
    int matrix_size = X.cols * X.rows;
    int total_threads = matrix_size;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    relu_kernel<<blocks_per_grid, threads_per_block>>(Y.data, X.data, matrix_size);

}


void matmul_gpu(const DeviceMatrix &out, DeviceMatrix& left, const DeviceMatrix& right){
    int total_threads = left.rows * right.cols;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    int M = left.rows;
    int K = left.cols;
    int N = right.cols;

    naive_matmul_kernel<<blocks_per_grid, threads_per_block>>(out.data, left.data, right.data, M, K, N);
}

int main(){




}
#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H
__global__ void naive_matmul_kernel(float *out, float *left, float *right, int M, int K, int N);
__global__ void tiled_matmul_kernel(float *out, float *left, float *right, int M, int K, int N);
#endif

#ifndef SOFTMAX_H
#define SOFTMAX_H
#define BLOCKSIZE 256
__global__ void softmax_kernel(float *out, float *in, int N, float *max, float *sum);
__global__ void max_kernel(float *out, float *in, int N);
__global__ void sum_kernel(float *out, float *in, int N, float *max);
__global__ void softmax_grad_kernel(float *grad, float *y, int N);
#endif
#ifndef SOFTMAX_H
#define SOFTMAX_H
#define BLOCKSIZE 256
__general__ void softmax_kernel(float *out, float *in, int N, float *max, float *sum);
__general__ void max_kernel(float *out, float *in, int N);
__general__ void sum_kernel(float *out, float *in, int N, float *max);
__global__ void softmax_grad_kernel(float *grad, float *y, int N);
#endif
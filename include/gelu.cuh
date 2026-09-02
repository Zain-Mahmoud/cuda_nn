#ifndef GELU_H
#define GELU_H
__global__ void gelu_kernel(float *out, float *in, int N);
__global__ void gelu_grad_kernel(float *out, float *in, int N);
#endif
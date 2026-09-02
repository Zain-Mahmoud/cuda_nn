#ifndef TANH_H
#define TANH_H
__global__ void tanh_kernel(float *out, float *in, int N);
__global__ void tanh_grad_kernel(float *out, float *in, int N);
#endif
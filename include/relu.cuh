#ifndef RELU_KERNEL_H
#define RELU_KERNEL_H

__global__ void relu_kernel(float *out, float *in, int N);
__global__ void relu_grad_kernel(float *out, float *in, int N);
__global__ void leaky_relu_kernel(float *out, float *in, int N, float alpha);
__global__ void leaky_relu_grad_kernel(float *out, float *in, int N, float alpha);

#endif
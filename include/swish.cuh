#ifndef SWISH_H
#define SWISH_H
float stable_sigmoid(float x);
__global__ void swish_kernel(float *out, float *in, int N);
__global__ void swish_grad_kernel(float *out, float *in, int N);

#endif
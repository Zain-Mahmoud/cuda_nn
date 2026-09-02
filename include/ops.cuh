#ifndef OPS_H
#define OPS_H
#include <device_matrix.hpp>

void relu_gpu(DeviceMatrix& X, DeviceMatrix& Y);
void relu_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y);

void matmul_gpu(DeviceMatrix &out, DeviceMatrix& left, DeviceMatrix& right);
void matmul_tiled_gpu(DeviceMatrix &out, DeviceMatrix& left, DeviceMatrix& right)

void softmax_gpu(DeviceMatrix &X, DeviceMatrix &Y);
void softmax_grad_gpu(DeviceMatrix &X, DeviceMatrix &Y);

void tanh_gpu(DeviceMatrix& X, DeviceMatrix& Y);
void tanh_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y);

void leaky_relu_gpu(DeviceMatrix& X, DeviceMatrix& Y, float alpha);
void leaky_relu_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y, float alpha);

void gelu_gpu(DeviceMatrix& X, DeviceMatrix& Y);
void gelu_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y);

void siwsh_gpu(DeviceMatrix& X, DeviceMatrix& Y);
void siwsh_grad_gpu(DeviceMatrix& X, DeviceMatrix& Y);

int next_power_of_two(int N);
#endif
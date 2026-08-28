#ifndef OPS_H
#define OPS_H
#include <device_matrix.hpp>
void matmul_gpu(const DeviceMatrix &out, DeviceMatrix& left, const DeviceMatrix& right);
void relu_gpu(const DeviceMatrix& X, DeviceMatrix& Y);
#endif
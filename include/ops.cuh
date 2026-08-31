#ifndef OPS_H
#define OPS_H
#include <device_matrix.hpp>
void matmul_gpu(DeviceMatrix &out, DeviceMatrix& left, DeviceMatrix& right);
void relu_gpu(DeviceMatrix& X, DeviceMatrix& Y);
#endif
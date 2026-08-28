#include <device_matrix.hpp>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;

DeviceMatrix::DeviceMatrix(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    cudaMalloc(&data, rows * cols * sizeof(float));
}

DeviceMatrix::~DeviceMatrix(){
    cudaFree(data);
}

void Device::copy_to_host(void *ptr){
    cudaMemcpy(data, ptr, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

void Device::copy_to_device(void *ptr){
    cudaMemcpy(ptr, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

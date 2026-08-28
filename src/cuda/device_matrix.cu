#include <device_matrix.hpp>
#include <cstdlib>
#include <cuda_runtime.h>
#include <matrix.hpp>
using namespace std;

DeviceMatrix::DeviceMatrix(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    cudaMalloc(&data, rows * cols * sizeof(float));
}

DeviceMatrix::~DeviceMatrix(){
    cudaFree(data);
}

void DeviceMatrix::copy_to_host(const Matrix &m){
    cudaMemcpy(data, m.data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

void DeviceMatrix::copy_to_device(const Matrix &m){
    cudaMemcpy(m.data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

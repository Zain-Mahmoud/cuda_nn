#include <device_matrix.hpp>
#include <cstdlib>
#include <cuda_runtime.h>
#include <matrix.hpp>
using namespace std;

DeviceMatrix::DeviceMatrix(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    cudaError_t err = cudaMalloc(&data, rows * cols * sizeof(float));

    if (err != cudaSuccess){
        cerr << "cudaMalloc for DeviceMatrix failed: "
         << cudaGetErrorString(err) << endl;
    }

}

DeviceMatrix::~DeviceMatrix(){
    cudaError_t err = cudaFree(data);

    if (err != cudaSuccess){
        cerr << "cudaFree for DeviceMatrix failed: "
         << cudaGetErrorString(err) << endl;
    }
}

void DeviceMatrix::copy_to_host(const Matrix &m){
    cudaError_t err = cudaMemcpy(data, m.data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        cerr << "cudaMemcpy copy_to_host failed: "
         << cudaGetErrorString(err) << endl;
    }
}

void DeviceMatrix::copy_to_device(const Matrix &m){
    cudaError_t err = cudaMemcpy(m.data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess){
        cerr << "cudaMemcpy copy_to_host failed: "
         << cudaGetErrorString(err) << endl;
    }
}

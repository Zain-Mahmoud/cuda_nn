#ifndef DEVICE_MATRIX_H
#define DEVICE_MATRIX_H
#include <matrix.hpp>
class DeviceMatrix{
    public:
        int rows, cols;
        float *data;
        
        DeviceMatrix(int, int);
        ~DeviceMatrix();

        void copy_to_host(const Matrix&);
        void copy_from_host(const Matrix&);

};
#endif
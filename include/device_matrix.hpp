#ifndef DEVICE_MATRIX_H
#define DEVICE_MATRIX_H
class DeviceMatrix{
    public:
        int rows, cols;
        float *data;
        DeviceMatrix(int, int);
        ~DeviceMatrix();

        void copy_to_host(void *);
        void copy_from_host(void *);


};
#endif
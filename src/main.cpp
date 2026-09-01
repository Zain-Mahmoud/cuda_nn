#include <iostream>
#include <matrix.hpp>
#include <device_matrix.hpp>
#include <ops.cuh>
#include <linear.hpp>
#include <relu.hpp>

using namespace std;

void fill(Matrix &X, float val){
    for (int i = 0; i < X.rows; i++){
        for (int j = 0; j < X.cols; j++){
            X.data[i * X.cols + j] = val;
        }
    }
}

void fill_alternating(Matrix &X, float val){

    for (int i = 0; i < X.rows; i++){
        for (int j = 0; j < X.cols; j++){
            val *= -1;
            X.data[i * X.cols + j] = val;
        }
    }
}

void mat_print(Matrix &X){

    for (int i = 0; i < X.rows; i++){
        for (int j = 0; j < X.cols; j++){
            cout << X.data[i * X.cols + j] << " ";
        }
        cout << endl;
    }
}


int main(){
    Matrix A(2,3); // initialize 2x3 matrix
    Matrix B(3,4);  // initialize 3x4 matrix
    Matrix C(2, 4);

    Matrix X(2, 2); // initialize 2x2 matrix
    Matrix Y(2, 2); // initialize 2x2 matrix

    fill(A, 5); // fill with 5s
    fill(B, 6); // fill with 6s

    fill_alternating(X, 3); // fill with 3, -3, 3, -3 etc
    mat_print(X);

    //initialize device matrices
    DeviceMatrix dA(2,3); 
    DeviceMatrix dB(3,4);
    DeviceMatrix dC(2, 4);
    
    DeviceMatrix dX(2, 2);
    DeviceMatrix dY(2,2);

    // copy values to device matrices
    dA.copy_to_device(A);
    dB.copy_to_device(B);
    dX.copy_to_device(X);

    relu_gpu(dX, dY);
    dY.copy_to_host(Y);

    matmul_gpu(dC, dA, dB);
    dC.copy_to_host(C);

    mat_print(Y);

}
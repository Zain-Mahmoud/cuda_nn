#include <matrix.hpp>
#include <new>
Matrix *relu_cpu(Matrix *X){
    Matrix *Y = new Matrix(X->rows, X->cols);
    for (int i = 0; i < X->rows; i++){
        for (int j = 0; j < X->cols; j++){
            if ((*X)(i, j) > 0){
                (*Y)(i, j) = (*X)(i, j);
            } else {
                (*Y)(i, j) = 0;
            }
        }
    }
    return Y;
}
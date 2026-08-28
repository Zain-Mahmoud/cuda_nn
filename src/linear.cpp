#include <matrix.hpp>
#include <cstdlib>
#include <new>
using namespace std;

Matrix *matmul_cpu(Matrix *A, Matrix *B){
    Matrix* C = new Matrix(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++){
        for (int j = 0; j < B->cols; j++){
            float sum = 0;
            for (int k = 0; k < A->cols; k++){
                sum += (*A)(i,k) * (*B)(k, j);
            }
            (*C)(i,j) = sum;
        }
    }
    return C;
}
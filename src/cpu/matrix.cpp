#include <matrix.hpp>
#include <cstdlib>
using namespace std;

Matrix::Matrix(int n_rows, int n_cols){
    rows = n_rows;
    cols = n_cols;
    data = (float *) malloc(sizeof(float) * n_rows * n_cols);
}

float& Matrix::operator()(int i, int j){
    return data[(i * rows) + j ];
}
Matrix::~Matrix(){
    free(data);
}
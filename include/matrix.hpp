#ifndef MATRIX_H
#define MATRIX_H
class Matrix {
    public:
        int rows;
        int cols;
        float *data;
        Matrix (int, int);
        ~Matrix();

        float& operator() (int, int);

};
#endif
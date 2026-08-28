#include <iostream>
#include <matrix.hpp>
#include <linear.hpp>
#include <relu.hpp>
using namespace std;

int main(){

    Matrix m(2, 2);
    m(0, 0)=1;
    m(0,1)=0.5;
    m(1,0)=-100;
    m(1,1)=20;
    Matrix *C = relu_cpu(&m);

    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            cout << (*C)(i,j) << endl;
        }
    }
    return 0;
}
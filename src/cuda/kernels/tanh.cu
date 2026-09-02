#include <cmath>
using namespace std;
__global__ void tanh_kernel(float *out, float *in, int N){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= N){
        return;
    }
    float x = in[idx];
    out[idx] = tanhf(x);

}
__global__ void tanh_grad_kernel(float *out, float *in, int N){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= N){
        return;
    }
    float x = in[idx];
    float tanh = tanhf(x);
    out[idx] = 1 - (tanh * tanh);

}
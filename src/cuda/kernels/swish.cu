#include <cmath>
using namespace std;
float stable_sigmoid(float x) {
    if (x >= 0) {
        float z = expf(-x);
        return 1.0 / (1.0 + z);
    } else {
        float z = expf(x);
        return z / (1.0 + z);
    }
}

__global__ void swish_kernel(float *out, float *in, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= N){
        return ;
    }
    float x = in[i];
    out[i] = x * stable_sigmoid(x);

}

__global__ void swish_grad_kernel(float *out, float *in, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= N){
        return ;
    }
    float x = in[i];
    float s = stable_sigmoid(x);
    out[i] = s + (s*x*(1-s));
}
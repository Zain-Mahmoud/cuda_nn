#include <cmath>
#include <numbers>
using namespace std;
__global__ void gelu_kernel(float *out, float *in, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float pi_f = numbers::pi_v<float>; 

    if (i >= N){
        return;
    } 
    float x = in[i];
    out[i] = 0.5*x*(1+tanhf((2/pi_f)*(x+0.044715*powf(x, 3))));

}
__global__ void gelu_grad_kernel(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    float x = in[idx];

    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;

    float x3 = x * x * x;

    float u = SQRT_2_OVER_PI * (x + COEFF * x3);
    float t = tanhf(u);

    float du = SQRT_2_OVER_PI *
               (1.0f + 3.0f * COEFF * x * x);

    out[idx] =
        0.5f * (1.0f + t)
        + 0.5f * x * (1.0f - t * t) * du;
}
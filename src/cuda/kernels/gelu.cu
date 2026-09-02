#include <gelu.cuh>
#include <cmath>

__global__ void gelu_kernel(float *out, float *in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;

    float x = in[i];
    float x2 = x * x;
    float x3 = x2 * x;

    float u = SQRT_2_OVER_PI * (x + COEFF * x3);

    out[i] = 0.5f * x * (1.0f + tanhf(u));
}


__global__ void gelu_grad_kernel(float *out, float *in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;

    float x = in[i];
    float x2 = x * x;
    float x3 = x2 * x;

    float u = SQRT_2_OVER_PI * (x + COEFF * x3);
    float t = tanhf(u);

    float du = SQRT_2_OVER_PI *
               (1.0f + 3.0f * COEFF * x2);

    out[i] =
        0.5f * (1.0f + t)
        + 0.5f * x * (1.0f - t * t) * du;
}
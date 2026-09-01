#include <cmath>
#include <softmax.cuh>
using namespace std;

__global__ void softmax_kernel(float *out, float *in, int N, float *max, float *sum){
    int idx = threadIdx.x;
    float vmax = *max; 
    float vsum = *sum;

    if (idx >= N){
        return;
    }

    out[idx] = expf(in[idx] - vmax) / vsum;
}

__global__ void max_kernel(float *out, const float *in, int N) {
    int idx = threadIdx.x;

    __shared__ float values[BLOCKSIZE];


    if (idx < N) {
        values[idx] = in[idx];
    } else {
        values[idx] = -INFINITY;
    }

    __syncthreads();

    // Parallel reduction.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {

        if (idx < stride) {
            values[idx] = max(values[idx],
                              values[idx + stride]);
        }

        __syncthreads();
    }

    if (idx == 0) {
        *out = values[0];
    }
}

__global__ void sum_kernel(float *out, const float *in, int N, float *max) {
    int idx = threadIdx.x;
    float vmax = *max;

    __shared__ float values[BLOCKSIZE];

    if (idx < N) {
        values[idx] = expf(in[idx] - vmax);
    } else {
        values[idx] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {

        if (idx < stride) {
            values[idx] += expf(values[idx + stride] - vmax);
        }

        __syncthreads();
    }

    if (idx == 0) {
        *out = values[0];
    }
}
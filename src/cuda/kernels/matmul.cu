#include <cmath>
#include <matmul.cuh>

using namespace std;
__global__ void naive_matmul_kernel(float *out, float *left, float *right, int M, int K, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx > M * N - 1){
        return;
    }

    int row = idx / N;
    int col = idx % N;
    float sum = 0.0f;
    for (int k = 0; k < K; k++){
        sum += left[k + (row * K)] * right[(k*N) + col];
    }
    out[idx] = sum;
}

__global__ void tiled_matmul_kernel(float *out, float *left, float *right, int M, int K, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int i = by*blockDim.y + ty;
    int j = bx*blockDim.x + tx;

    __shared__ float sh_left[TILE_SIZE][TILE_SIZE]; 
    __shared__ float sh_right[TILE_SIZE][TILE_SIZE];
    float sum = 0;

    for (int phase = 0; phase < ceil((float)K / TILE_SIZE); phase++){

        if (i < M && (phase * TILE_SIZE + tx < K)){
            sh_left[ty][tx] = left[i * K + phase * TILE_SIZE +tx];
        } else {
            sh_left[ty][tx] = 0.0f;
        }

        if (j < N && (phase * TILE_SIZE + ty < K)){
            sh_right[ty][tx] = right[j + (phase * TILE_SIZE + ty)*N];
        } else {
            sh_right[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++){
            sum += sh_left[ty][k] * sh_right[k][tx];
        }

        __syncthreads();
    }
    if (i < M && j < N){
            out[i * N + j] = sum;
    }
}
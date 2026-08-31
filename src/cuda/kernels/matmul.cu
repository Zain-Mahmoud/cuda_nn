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

}
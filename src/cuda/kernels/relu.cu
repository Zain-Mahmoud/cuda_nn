__global__ void relu_kernel(float *out, float *in, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= N){
        return;
    }
    if (in[i] > 0){
        out[i] = in[i];
    } else {
        out[i] = 0.0f;
    }
}
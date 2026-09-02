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
__global__ void relu_grad_kernel(float *out, float *in, int N){
    
    if (i >= N){
        return;
    }
    if (in[i] > 0){
        out[i] = 1.0f;
    } else {
        out[i] = 0.0f;
    }

}

__global__ void leaky_relu_kernel(float *out, float *in, int N, float alpha){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= N){
        return;
    }
    if (in[i] > 0){
        out[i] = in[i];
    } else {
        out[i] = alpha * in[i];
    }
}


__global__ void leaky_relu_grad_kernel(float *out, float *in, int N, float alpha){
    
    if (i >= N){
        return;
    }
    if (in[i] > 0){
        out[i] = 1.0f;
    } else {
        out[i] = alpha;
    }

}
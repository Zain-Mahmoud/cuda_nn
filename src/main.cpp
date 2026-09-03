#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cassert>

#include <matrix.hpp>
#include <device_matrix.hpp>
#include <ops.cuh>

using namespace std;

void cpu_relu(const Matrix& X, Matrix& Y) {
    int N = X.rows * X.cols;

    for (int i = 0; i < N; i++) {
        Y.data[i] = max(0.0f, X.data[i]);
    }
}

void cpu_relu_grad(const Matrix& X, Matrix& Y) {
    int N = X.rows * X.cols;

    for (int i = 0; i < N; i++) {
        Y.data[i] = X.data[i] > 0.0f ? 1.0f : 0.0f;
    }
}

void cpu_leaky_relu(const Matrix& X, Matrix& Y, float alpha) {
    int N = X.rows * X.cols;

    for (int i = 0; i < N; i++) {
        Y.data[i] = X.data[i] > 0.0f
                  ? X.data[i]
                  : alpha * X.data[i];
    }
}

void cpu_leaky_relu_grad(const Matrix& X, Matrix& Y, float alpha) {
    int N = X.rows * X.cols;

    for (int i = 0; i < N; i++) {
        Y.data[i] = X.data[i] > 0.0f ? 1.0f : alpha;
    }
}

void cpu_tanh(const Matrix& X, Matrix& Y) {
    int N = X.rows * X.cols;

    for (int i = 0; i < N; i++) {
        Y.data[i] = tanhf(X.data[i]);
    }
}

void cpu_tanh_grad(const Matrix& X, Matrix& Y) {
    int N = X.rows * X.cols;

    for (int i = 0; i < N; i++) {
        float t = tanhf(X.data[i]);
        Y.data[i] = 1.0f - t * t;
    }
}

void cpu_gelu(const Matrix& X, Matrix& Y) {
    int N = X.rows * X.cols;

    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;

    for (int i = 0; i < N; i++) {
        float x = X.data[i];
        float x3 = x * x * x;

        float u = SQRT_2_OVER_PI * (x + COEFF * x3);

        Y.data[i] =
            0.5f * x * (1.0f + tanhf(u));
    }
}

void cpu_gelu_grad(const Matrix& X, Matrix& Y) {
    int N = X.rows * X.cols;

    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;

    for (int i = 0; i < N; i++) {
        float x = X.data[i];

        float x2 = x * x;
        float x3 = x2 * x;

        float u = SQRT_2_OVER_PI * (x + COEFF * x3);
        float t = tanhf(u);

        float du =
            SQRT_2_OVER_PI *
            (1.0f + 3.0f * COEFF * x2);

        Y.data[i] =
            0.5f * (1.0f + t)
            + 0.5f * x * (1.0f - t * t) * du;
    }
}

float cpu_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

void cpu_swish(const Matrix& X, Matrix& Y) {
    int N = X.rows * X.cols;

    for (int i = 0; i < N; i++) {
        float x = X.data[i];
        float s = cpu_sigmoid(x);

        Y.data[i] = x * s;
    }
}

void cpu_swish_grad(const Matrix& X, Matrix& Y) {
    int N = X.rows * X.cols;

    for (int i = 0; i < N; i++) {
        float x = X.data[i];
        float s = cpu_sigmoid(x);

        Y.data[i] = s + x * s * (1.0f - s);
    }
}


void cpu_matmul(
    const Matrix& A,
    const Matrix& B,
    Matrix& C
) {
    assert(A.cols == B.rows);
    assert(C.rows == A.rows);
    assert(C.cols == B.cols);

    int M = A.rows;
    int K = A.cols;
    int N = B.cols;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                sum +=
                    A.data[i * K + k] *
                    B.data[k * N + j];
            }

            C.data[i * N + j] = sum;
        }
    }
}

void cpu_softmax(const Matrix& X, Matrix& Y) {
    assert(X.cols == 1);
    assert(Y.cols == 1);

    int N = X.rows;

    float max_val = X.data[0];

    for (int i = 1; i < N; i++) {
        max_val = max(max_val, X.data[i]);
    }

    float sum = 0.0f;

    for (int i = 0; i < N; i++) {
        sum += expf(X.data[i] - max_val);
    }

    for (int i = 0; i < N; i++) {
        Y.data[i] =
            expf(X.data[i] - max_val) / sum;
    }
}


void cpu_softmax_grad(const Matrix& X, Matrix& Y) {
    assert(X.cols == 1);

    int N = X.rows;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            float delta = (i == j) ? 1.0f : 0.0f;

            Y.data[i * N + j] =
                X.data[i] *
                (delta - X.data[j]);
        }
    }
}


void random_fill(Matrix& X, float low = -3.0f, float high = 3.0f) {

    static random_device rd;
    static mt19937 gen(rd());

    uniform_real_distribution<float> dist(low, high);

    int N = X.rows * X.cols;

    for (int i = 0; i < N; i++) {
        X.data[i] = dist(gen);
    }
}

bool compare_matrices(
    const Matrix& A,
    const Matrix& B,
    float tolerance = 1e-4f
) {
    if (A.rows != B.rows || A.cols != B.cols) {
        return false;
    }

    int N = A.rows * A.cols;

    for (int i = 0; i < N; i++) {

        float diff = fabs(A.data[i] - B.data[i]);

        if (diff > tolerance) {
            cerr << "Mismatch at index " << i
                 << ": CPU = " << A.data[i]
                 << ", GPU = " << B.data[i]
                 << ", diff = " << diff << '\n';

            return false;
        }
    }

    return true;
}

void print_test_result(const string& name, bool passed) {
    cout << left << setw(25)
         << name
         << (passed ? "PASS" : "FAIL")
         << '\n';
}

template <typename Func>
float benchmark_gpu(Func func, int iterations = 100) {

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 10; i++) {
        func();
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++) {
        func();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iterations;
}


template <typename Func>
double benchmark_cpu(Func func, int iterations = 10) {

    // Warmup
    for (int i = 0; i < 2; i++) {
        func();
    }

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        func();
    }

    auto end = chrono::high_resolution_clock::now();

    double milliseconds =
        chrono::duration<double, milli>(
            end - start
        ).count();

    return milliseconds / iterations;
}

int main() {

    cout << fixed << setprecision(6);

    cout << "\n=== ACTIVATION CORRECTNESS ===\n\n";

    const int R = 128;
    const int C = 128;

    Matrix X(R, C);

    Matrix cpu_Y(R, C);
    Matrix gpu_Y(R, C);

    random_fill(X);

    DeviceMatrix dX(R, C);
    DeviceMatrix dY(R, C);

    dX.copy_to_device(X);

    cpu_relu(X, cpu_Y);

    relu_gpu(dX, dY);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "ReLU",
        compare_matrices(cpu_Y, gpu_Y)
    );

    cpu_relu_grad(X, cpu_Y);

    relu_grad_gpu(dX, dY);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "ReLU gradient",
        compare_matrices(cpu_Y, gpu_Y)
    );

    cpu_tanh(X, cpu_Y);

    tanh_gpu(dX, dY);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "Tanh",
        compare_matrices(cpu_Y, gpu_Y)
    );


    cpu_tanh_grad(X, cpu_Y);

    tanh_grad_gpu(dX, dY);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "Tanh gradient",
        compare_matrices(cpu_Y, gpu_Y)
    );


    cpu_gelu(X, cpu_Y);

    gelu_gpu(dX, dY);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "GELU",
        compare_matrices(cpu_Y, gpu_Y)
    );

    cpu_gelu_grad(X, cpu_Y);

    gelu_grad_gpu(dX, dY);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "GELU gradient",
        compare_matrices(cpu_Y, gpu_Y)
    );

    cpu_swish(X, cpu_Y);

    siwsh_gpu(dX, dY);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "Swish",
        compare_matrices(cpu_Y, gpu_Y)
    );

    cpu_swish_grad(X, cpu_Y);

    siwsh_grad_gpu(dX, dY);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "Swish gradient",
        compare_matrices(cpu_Y, gpu_Y)
    );

    float alpha = 0.01f;

    cpu_leaky_relu(X, cpu_Y, alpha);

    leaky_relu_gpu(dX, dY, alpha);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "Leaky ReLU",
        compare_matrices(cpu_Y, gpu_Y)
    );

    cpu_leaky_relu_grad(X, cpu_Y, alpha);

    leaky_relu_grad_gpu(dX, dY, alpha);
    cudaDeviceSynchronize();
    dY.copy_to_host(gpu_Y);

    print_test_result(
        "Leaky ReLU gradient",
        compare_matrices(cpu_Y, gpu_Y)
    );


    cout << "\n=== MATMUL CORRECTNESS ===\n\n";

    int M = 128;
    int K = 128;
    int N = 128;

    Matrix A(M, K);
    Matrix B(K, N);

    Matrix cpu_C(M, N);
    Matrix gpu_C(M, N);

    random_fill(A);
    random_fill(B);

    DeviceMatrix dA(M, K);
    DeviceMatrix dB(K, N);
    DeviceMatrix dC(M, N);

    dA.copy_to_device(A);
    dB.copy_to_device(B);

    cpu_matmul(A, B, cpu_C);

    matmul_gpu(dC, dA, dB);
    cudaDeviceSynchronize();

    dC.copy_to_host(gpu_C);

    print_test_result(
        "Naive matmul",
        compare_matrices(cpu_C, gpu_C, 1e-3f)
    );


    matmul_tiled_gpu(dC, dA, dB);
    cudaDeviceSynchronize();

    dC.copy_to_host(gpu_C);

    print_test_result(
        "Tiled matmul",
        compare_matrices(cpu_C, gpu_C, 1e-3f)
    );


    cout << "\n=== SOFTMAX CORRECTNESS ===\n\n";

    int softmax_N = 128;

    Matrix SX(softmax_N, 1);
    Matrix cpu_SY(softmax_N, 1);
    Matrix gpu_SY(softmax_N, 1);

    random_fill(SX, -10.0f, 10.0f);

    DeviceMatrix dSX(softmax_N, 1);
    DeviceMatrix dSY(softmax_N, 1);

    dSX.copy_to_device(SX);

    cpu_softmax(SX, cpu_SY);

    softmax_gpu(dSX, dSY);
    cudaDeviceSynchronize();

    dSY.copy_to_host(gpu_SY);

    print_test_result(
        "Softmax",
        compare_matrices(cpu_SY, gpu_SY, 1e-5f)
    );

    cout << "\n=== SOFTMAX GRADIENT ===\n\n";

    Matrix cpu_J(softmax_N, softmax_N);
    Matrix gpu_J(softmax_N, softmax_N);

    DeviceMatrix dJ(softmax_N, softmax_N);

    cpu_softmax_grad(cpu_SY, cpu_J);
    softmax_grad_gpu(dSY, dJ);
    cudaDeviceSynchronize();

    dJ.copy_to_host(gpu_J);

    print_test_result(
        "Softmax Jacobian",
        compare_matrices(cpu_J, gpu_J, 1e-5f)
    );
    cout << "\n=== PERFORMANCE ===\n\n";

    int BENCH_R = 1024;
    int BENCH_C = 1024;

    Matrix BX(BENCH_R, BENCH_C);
    Matrix BCpu(BENCH_R, BENCH_C);
    Matrix BGpu(BENCH_R, BENCH_C);

    random_fill(BX);

    DeviceMatrix dBX(BENCH_R, BENCH_C);
    DeviceMatrix dBGpu(BENCH_R, BENCH_C);

    dBX.copy_to_device(BX);

    double cpu_relu_ms = benchmark_cpu(
        [&]() {
            cpu_relu(BX, BCpu);
        }
    );

    float gpu_relu_ms = benchmark_gpu(
        [&]() {
            relu_gpu(dBX, dBGpu);
        }
    );

    cout << "ReLU\n";
    cout << "  CPU: " << cpu_relu_ms << " ms\n";
    cout << "  GPU: " << gpu_relu_ms << " ms\n";
    cout << "  Speedup: "
         << cpu_relu_ms / gpu_relu_ms
         << "x\n\n";


    double cpu_tanh_ms = benchmark_cpu(
        [&]() {
            cpu_tanh(BX, BCpu);
        }
    );

    float gpu_tanh_ms = benchmark_gpu(
        [&]() {
            tanh_gpu(dBX, dBGpu);
        }
    );

    cout << "Tanh\n";
    cout << "  CPU: " << cpu_tanh_ms << " ms\n";
    cout << "  GPU: " << gpu_tanh_ms << " ms\n";
    cout << "  Speedup: "
         << cpu_tanh_ms / gpu_tanh_ms
         << "x\n\n";

    double cpu_gelu_ms = benchmark_cpu(
        [&]() {
            cpu_gelu(BX, BCpu);
        }
    );

    float gpu_gelu_ms = benchmark_gpu(
        [&]() {
            gelu_gpu(dBX, dBGpu);
        }
    );

    cout << "GELU\n";
    cout << "  CPU: " << cpu_gelu_ms << " ms\n";
    cout << "  GPU: " << gpu_gelu_ms << " ms\n";
    cout << "  Speedup: "
         << cpu_gelu_ms / gpu_gelu_ms
         << "x\n\n";

    double cpu_swish_ms = benchmark_cpu(
        [&]() {
            cpu_swish(BX, BCpu);
        }
    );

    float gpu_swish_ms = benchmark_gpu(
        [&]() {
            siwsh_gpu(dBX, dBGpu);
        }
    );

    cout << "Swish\n";
    cout << "  CPU: " << cpu_swish_ms << " ms\n";
    cout << "  GPU: " << gpu_swish_ms << " ms\n";
    cout << "  Speedup: "
         << cpu_swish_ms / gpu_swish_ms
         << "x\n\n";

    cout << "=== MATMUL PERFORMANCE ===\n\n";

    int BM = 1024;
    int BK = 1024;
    int BN = 1024;

    Matrix BA(BM, BK);
    Matrix BB(BK, BN);

    Matrix BCPU(BM, BN);
    Matrix BGPU(BM, BN);

    random_fill(BA);
    random_fill(BB);

    DeviceMatrix dBA(BM, BK);
    DeviceMatrix dBB(BK, BN);
    DeviceMatrix dBC(BM, BN);

    dBA.copy_to_device(BA);
    dBB.copy_to_device(BB);


    double cpu_matmul_ms = benchmark_cpu(
        [&]() {
            cpu_matmul(BA, BB, BCPU);
        },
        3
    );

    float gpu_naive_ms = benchmark_gpu(
        [&]() {
            matmul_gpu(dBC, dBA, dBB);
        },
        20
    );

    float gpu_tiled_ms = benchmark_gpu(
        [&]() {
            matmul_tiled_gpu(dBC, dBA, dBB);
        },
        20
    );


    cout << "Matrix multiplication: "
         << BM << " x " << BK
         << " * "
         << BK << " x " << BN
         << "\n\n";

    cout << "CPU:\n";
    cout << "  " << cpu_matmul_ms << " ms\n\n";

    cout << "GPU naive:\n";
    cout << "  " << gpu_naive_ms << " ms\n";
    cout << "  Speedup: "
         << cpu_matmul_ms / gpu_naive_ms
         << "x\n\n";

    cout << "GPU tiled:\n";
    cout << "  " << gpu_tiled_ms << " ms\n";
    cout << "  Speedup: "
         << cpu_matmul_ms / gpu_tiled_ms
         << "x\n";

    cout << "\n";
    cout << "Tiled vs naive: "
         << gpu_naive_ms / gpu_tiled_ms
         << "x\n";


    cudaDeviceSynchronize();

    return 0;
}
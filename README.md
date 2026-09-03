# CUDA Deep Learning

A from-scratch neural-network computation library written in **C++ and CUDA**, designed to explore GPU programming, CUDA kernel optimization, memory management, parallel reductions, and CPU/GPU performance differences.

The project implements common neural-network operations on both the CPU and GPU, with correctness tests comparing the two implementations.

## Features

### Activation Functions

CPU and CUDA implementations of:

* ReLU
* Leaky ReLU
* Tanh
* GELU
* Swish

Each activation also includes its corresponding gradient.

### Matrix Operations

* CPU matrix multiplication
* Naive CUDA matrix multiplication
* Tiled CUDA matrix multiplication using shared memory
* Host ↔ device matrix transfers

### Softmax

* Numerically stable softmax
* GPU max reduction
* GPU sum reduction
* Softmax Jacobian
* CPU/GPU correctness comparison

The softmax implementation uses the numerically stable formulation:

$$\mathrm{softmax}(x_i) = \frac{e^{x_i-\max(x)}}{\sum_j e^{x_j-\max(x)}} $$

Subtracting the maximum prevents large positive logits from causing exponential overflow.

### GPU Techniques

The project demonstrates several fundamental CUDA concepts:

* CUDA kernels
* Thread and block indexing
* 1D and 2D grid configurations
* Global memory
* Shared memory
* Thread synchronization
* Parallel reduction
* Tiled matrix multiplication
* CUDA events for GPU benchmarking
* Host/device memory transfers
* CUDA error checking

---

## Project Structure

```text
cuda_deep_learning
├── include
│   ├── device_matrix.hpp
│   ├── gelu.cuh
│   ├── matmul.cuh
│   ├── matrix.hpp
│   ├── ops.cuh
│   ├── relu.cuh
│   ├── softmax.cuh
│   ├── swish.cuh
│   └── tanh.cuh
├── src
│    ├── cpu
│    │   └── matrix.cpp
│    └── cuda
│       ├── kernels
│       │   ├── gelu.cu
│       │   ├── matmul.cu
│       │   ├── relu.cu
│       │   ├── softmax.cu
│       │   ├── swish.cu
│       │   └── tanh.cu
│       ├── device_matrix.cu
│       ├── main.cu
│       └── ops.cu
├── README.md
└── CMakeLists.txt
```

### Design

The project separates the public API, CUDA kernel declarations, kernel implementations, and higher-level GPU operation wrappers.

```text
main.cu
    │
    ▼
ops.cuh
    │
    ▼
ops.cu
    │
    ├── relu.cu
    ├── tanh.cu
    ├── gelu.cu
    ├── swish.cu
    ├── softmax.cu
    └── matmul.cu
```

The `.cuh` files contain CUDA-specific declarations, while `.cu` files contain the implementations.

`ops.cu` provides the higher-level functions used by the rest of the application, such as:

```cpp
relu_gpu(X, Y);
matmul_gpu(out, left, right);
matmul_tiled_gpu(out, left, right);
softmax_gpu(X, Y);
```

---

## Requirements

* CMake >= 3.20
* C++17
* CUDA Toolkit
* NVIDIA GPU with CUDA support
* NVIDIA driver compatible with the installed CUDA Toolkit

The project requires a CUDA compiler (`nvcc`) to build the GPU components.

Check your CUDA installation with:

```bash
nvcc --version
```

---

## Building

Clone the repository and create a build directory:

```bash
git clone <repository-url>
cd cuda_nn

mkdir build
cd build
```

Configure the project:

```bash
cmake ..
```

Build:

```bash
cmake --build . -j
```

Run:

```bash
./cuda_nn
```

If CMake cannot locate CUDA, verify that the CUDA Toolkit is installed and that `nvcc` is available on your `PATH`.

---

## Implementation Details

### ReLU

Each GPU thread processes one matrix element:

```text
Input
  │
  ▼
x ──► max(0, x)
  │
  ▼
Output
```

This is an example of an **element-wise GPU operation**, where each output element can be computed independently.

---

### Matrix Multiplication

The naive implementation assigns one CUDA thread to each output element.

For:

$$
C = AB
$$

each thread computes one:

$$
C_{ij} = \sum_k A_{ik}B_{kj}
$$

The mapping is:

```text
threadIdx/blockIdx
        │
        ▼
   output (row, col)
        │
        ▼
   dot product
        │
        ▼
      C[row][col]
```

#### Tiled Matrix Multiplication

The tiled implementation uses CUDA shared memory to reduce repeated global-memory accesses.

```text
Global Memory
     │
     ▼
┌─────────────┐
│ Tile of A   │
│             │
└─────────────┘
       │
       ▼
 Shared Memory

┌─────────────┐
│ Tile of B   │
│             │
└─────────────┘
       │
       ▼
 Shared Memory
       │
       ▼
   Dot Products
       │
       ▼
      C
```

A 2D CUDA block corresponds to a tile of the output matrix.

---

### Softmax

Softmax requires two reductions:

1. Find the maximum value.
2. Compute the sum of exponentials after subtracting the maximum.

The GPU implementation therefore performs:

```text
X
│
├──► max reduction ──► max(X)
│
├──► exp(X - max)
│
└──► sum reduction ──► Σ exp(X - max)
                           │
                           ▼
                    divide each value
                           │
                           ▼
                        Softmax
```

The reductions use shared memory and synchronized threads.

For arbitrary input sizes, the number of threads is rounded up to a power of two. Values outside the input range are padded with the appropriate identity value:

* `-∞` for maximum reduction
* `0` for sum reduction

---

### Softmax Jacobian

For a softmax output \(y\), the Jacobian is:

$$ J_{ij} = y_i(\delta_{ij}-y_j) $$

where:

$$
\delta_{ij} =
\begin{cases}
1 & i=j\\
0 & i\ne j
\end{cases}
$$

The implementation treats the already-computed softmax output as the input to the gradient operation:

```text
logits
   │
   ▼
softmax
   │
   ▼
probabilities y
   │
   ▼
softmax_grad
   │
   ▼
Jacobian J
```

This keeps the softmax operation and its derivative as separate operations.

---

## Correctness Testing

The executable includes CPU/GPU correctness tests.

For each operation, the GPU result is copied back to the host and compared against the CPU implementation.

Example:

```text
CPU implementation
        │
        ▼
     CPU result
        │
        │ compare
        ▼
     GPU result
        ▲
        │
GPU implementation
```

Floating-point comparisons use a tolerance rather than requiring exact equality.

Example output:

```text
=== ACTIVATION CORRECTNESS ===

ReLU                     PASS
Tanh                     PASS
GELU                     PASS
Swish                    PASS

=== MATRIX MULTIPLICATION ===

Naive Matmul             PASS
Tiled Matmul             PASS

=== SOFTMAX CORRECTNESS ===

Softmax                   PASS

=== SOFTMAX GRADIENT ===

Softmax Jacobian          PASS
```

---

## Benchmarking

The project also compares CPU and GPU execution time.

GPU benchmarks use **CUDA events** rather than CPU wall-clock timers. This is important because CUDA kernel launches are asynchronous with respect to the CPU.

The benchmark roughly follows:

```text
GPU operation
     │
     ▼
CUDA event ──► kernel ──► CUDA event
                            │
                            ▼
                       elapsed time
```

GPU benchmarks keep data on the device so that host/device transfer time does not dominate the kernel measurements.

The matrix multiplication benchmark compares:

* CPU matrix multiplication
* Naive CUDA matrix multiplication
* Tiled CUDA matrix multiplication

The tiled implementation is expected to outperform the naive implementation on sufficiently large matrices because it makes better use of shared memory and reduces redundant global-memory accesses.

---

## CUDA Error Checking

CUDA kernel launches are checked using:

```cpp
cudaGetLastError();
```

For debugging execution errors, synchronization can be used:

```cpp
cudaDeviceSynchronize();
```

These serve different purposes:

* `cudaGetLastError()` detects launch/configuration errors.
* `cudaDeviceSynchronize()` waits for the GPU and can surface errors that occurred during kernel execution.

---

## Future Improvements

Potential extensions include:

* [ ] Batched matrix multiplication
* [ ] More optimized reduction kernels
* [ ] Warp-level primitives
* [ ] cuBLAS comparison
* [ ] CUDA streams
* [ ] Pinned host memory
* [ ] Tensor Core / mixed-precision implementations
* [ ] Fused activation kernels
* [ ] GPU neural-network layers
* [ ] Full forward/backward neural-network training
* [ ] Additional performance profiling with Nsight Systems/Compute

---

## Motivation

This project is primarily an exploration of **GPU programming and CUDA performance**, rather than a production-ready neural-network framework.

The goal is to understand what happens underneath high-level ML frameworks by implementing common neural-network operations directly with CUDA:

```text
High-level ML operation
        │
        ▼
CUDA kernel
        │
        ▼
Threads / blocks
        │
        ▼
Global memory
Shared memory
Registers
        │
        ▼
GPU computation
```

By implementing both CPU and GPU versions, the project makes it possible to directly compare correctness, execution models, and performance.

## License

This project is for educational and experimental purposes.

// Toy kernel so the whole build chain can be exercised; replace with the real kernel.
// Keep this file free of torch/ATen includes: that is what lets `./optk compile` check it locally.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void scale_kernel(const float* __restrict__ x, float* __restrict__ y, float alpha, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)   // grid-stride loop
        y[i] = alpha * x[i];
}

void launch_scale(const float* x, float* y, float alpha, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 132 * 8);   // H100 has 132 SMs
    scale_kernel<<<blocks, threads, 0, stream>>>(x, y, alpha, n);
}

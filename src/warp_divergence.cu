// src/warp_divergence.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256
#define ITERATIONS 50000  // Increased for measurable time

// Control Path: All threads execute the SAME branch
__global__ void uniformKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 0.0f;
        // ALL threads do this loop
        for (int i = 0; i < ITERATIONS; i++) {
            val += sinf(i * 0.01f);
        }
        data[idx] = val;
    }
}

// Divergent Path: Threads split into two groups (Even vs Odd)
__global__ void divergentKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 0.0f;
        // Warp must serialize: Even threads wait for Odd, then Odd wait for Even
        if (threadIdx.x % 2 == 0) {
            for (int i = 0; i < ITERATIONS; i++) {
                val += sinf(i * 0.01f);
            }
        } else {
            for (int i = 0; i < ITERATIONS; i++) {
                val += sinf(i * 0.01f); // SAME WORK, different control path
            }
        }
        data[idx] = val;
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    uniformKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, n);

    // 1. Uniform (No Divergence)
    cudaEventRecord(start);
    for(int i=0; i<10; i++) // Run 10 times to average
        uniformKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_uniform = 0;
    cudaEventElapsedTime(&ms_uniform, start, stop);

    // 2. Divergent
    cudaEventRecord(start);
    for(int i=0; i<10; i++)
        divergentKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_divergent = 0;
    cudaEventElapsedTime(&ms_divergent, start, stop);

    printf("Warp Divergence Benchmark Results:\n");
    printf("Uniform Time: %.4f ms\n", ms_uniform);
    printf("Divergent Time: %.4f ms\n", ms_divergent);
    printf("Performance Degradation: %.2fx\n", ms_divergent / ms_uniform);

    FILE *f = fopen("results/warp_divergence.csv", "w");
    fprintf(f, "Mode,Time_ms\n");
    fprintf(f, "Uniform,%.4f\n", ms_uniform);
    fprintf(f, "Divergent,%.4f\n", ms_divergent);
    fclose(f);

    cudaFree(d_data);
    return 0;
}

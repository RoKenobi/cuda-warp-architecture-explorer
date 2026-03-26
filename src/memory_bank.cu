// src/memory_bank.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define ITERATIONS 100000  // Increased for measurable time

__global__ void sharedMemoryKernel(int stride) {
    // volatile forces actual memory access, prevents register caching
    volatile __shared__ float shared_data[BLOCK_SIZE * 32];
    int idx = threadIdx.x;
    float val = 0.0f;

    for (int i = 0; i < ITERATIONS; i++) {
        // Access pattern: idx * stride
        // Stride 1 = Bank 0, 1, 2... (No Conflict)
        // Stride 32 = Bank 0, 0, 0... (Max Conflict)
        shared_data[idx * stride] += 1.0f;
        val += shared_data[idx * stride]; // Prevent optimization
    }
    // Ensure writeback
    shared_data[idx] = val;
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    FILE *f = fopen("results/memory_bank.csv", "w");
    fprintf(f, "Stride,Time_ms\n");

    int strides[] = {1, 2, 4, 8, 16, 32};
    for (int s : strides) {
        cudaEventRecord(start);
        for(int i=0; i<10; i++) // Run 10 times to average
            sharedMemoryKernel<<<1, BLOCK_SIZE>>>(s);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Stride %d: %.4f ms\n", s, ms);
        fprintf(f, "%d,%.4f\n", s, ms);
    }
    fclose(f);
    return 0;
}

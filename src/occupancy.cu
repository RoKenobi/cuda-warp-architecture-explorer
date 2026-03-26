// src/occupancy.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dummyKernel() {
    // Heavy register usage to limit occupancy
    float r1, r2, r3, r4, r5, r6, r7, r8;
    for (int i = 0; i < 1000; i++) {
        r1 += sinf(i); r2 += cosf(i);
        r3 += sinf(i); r4 += cosf(i);
        r5 += sinf(i); r6 += cosf(i);
        r7 += sinf(i); r8 += cosf(i);
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    FILE *f = fopen("results/occupancy.csv", "w");
    fprintf(f, "BlockSize,Time_ms\n");

    int blocks[] = {128, 256, 512, 1024};
    for (int b : blocks) {
        cudaEventRecord(start);
        dummyKernel<<<1000, b>>>();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Block Size %d: %.4f ms\n", b, ms);
        fprintf(f, "%d,%.4f\n", b, ms);
    }
    fclose(f);
    return 0;
}

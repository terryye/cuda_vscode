#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void kernel() {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Thread %d is running\n", i);

}

int main() {
    kernel << <2, 4 >> > ();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Test Pass! \n");
    return 0;
}
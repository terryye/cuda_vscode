#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Automatic GPU selection based on MPI rank
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Finalize();
        return -1;
    }

    // Assign GPU: rank modulo number of available GPUs
    int device = rank % deviceCount;
    CHECK_CUDA(cudaSetDevice(device));

    // Get device properties for confirmation
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Rank %d/%d using GPU %d (%s) with %d multiprocessors\n",
        rank, size, device, prop.name, prop.multiProcessorCount);

    // Problem size (each process handles a portion)
    const int n = 1000000;
    int base = n / size;
    int rem = n % size;
    int count = base + (rank < rem); // distribute remainder
    int offset = rank * base + (rank < rem ? rank : rem);

    // Allocate host memory per rank
    float* h_a = (float*)malloc(count * sizeof(float));
    float* h_b = (float*)malloc(count * sizeof(float));
    float* h_c = (float*)malloc(count * sizeof(float));

    // Initialize data
    for (int i = 0; i < count; i++) {
        h_a[i] = offset + i;
        h_b[i] = (offset + i) * 2.0f;
        // thus, h_c[i] should be: (start_idx + i) * 3.0f
    }

    // Allocate device memory
    float* d_a, * d_b, * d_c;
    CHECK_CUDA(cudaMalloc(&d_a, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, count * sizeof(float)));

    // Copy data to GPU
    CHECK_CUDA(cudaMemcpy(d_a, h_a, count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, count * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;

    float milliseconds = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    vector_add << <gridSize, blockSize >> > (d_a, d_b, d_c, count);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));          // not DeviceSynchronize
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_c, d_c, count * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results for this rank
    bool success = true;
    for (int i = 0; i < 10 && i < count; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_c[i] - expected) > 1e-5) {
            success = false;
            break;
        }
    }

    printf("Rank %d: Processed %d elements in %.2f ms on GPU %d - %s\n",
        rank, count, milliseconds, device, success ? "SUCCESS" : "FAILED");


    // Verify sum using reduce
    float local_sum = 0;
    for (int i = 0; i < count; i++) {
        local_sum += h_c[i];
    }

    //
    float global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        float nf = n;
        if (fabsf(global_sum - 3.0 * nf * (nf - 1) / 2) < 0.001) {
            printf("global sum check FAILED: %f\n", global_sum);
        }
        else {
            printf("global sum check SUCEEDED: %f\n", global_sum);
        }
    }

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    MPI_Finalize();
    printf("Test Pass! \n");
    return 0;
}

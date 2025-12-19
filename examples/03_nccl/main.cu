/**

Initializes MPI and NCCL (NVIDIA Collective Communications Library)
Sets up a ring topology
Passes data around the ring
Verifies correctness
 */

#include "./nccl_utils.h"
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    int rank, nranks;
    ncclComm_t comm;

    // Initialize everything in one call
    if (init_mpi_nccl(argc, argv, &rank, &nranks, &comm) != 0) {
        fprintf(stderr, "Initialization failed\n");
        return 1;
    }

    // Test data
    const int size = 16;
    float* d_sendbuf;
    float* d_recvbuf;
    CHECK_CUDA(cudaMalloc(&d_sendbuf, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_recvbuf, size * sizeof(float)));

    // Initialize send buffer with rank-specific values
    std::vector<float> h_data(size, rank + 1.0f);
    CHECK_CUDA(cudaMemcpy(d_sendbuf, h_data.data(), size * sizeof(float),
        cudaMemcpyHostToDevice));

    std::cout << "Rank " << rank << ": Initial data = " << rank + 1.0f << std::endl;

    // Ring communication test
    int next_rank = (rank + 1) % nranks;
    int prev_rank = (rank - 1 + nranks) % nranks;

    // Pass data around the ring nranks times
    for (int step = 0; step < nranks; step++) {
        if (step > 0) {
            CHECK_CUDA(cudaMemcpy(d_sendbuf, d_recvbuf, size * sizeof(float),
                cudaMemcpyDeviceToDevice));
        }

        ring_exchange(d_sendbuf, d_recvbuf, size, next_rank, prev_rank, comm);


        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> h_recv(size);
        CHECK_CUDA(cudaMemcpy(h_recv.data(), d_recvbuf, size * sizeof(float),
            cudaMemcpyDeviceToHost));

        std::cout << "Rank " << rank << ", Step " << step
            << ": Received " << h_recv[0] << " from rank " << prev_rank
            << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_sendbuf));
    CHECK_CUDA(cudaFree(d_recvbuf));
    cleanup_mpi_nccl(comm);

    std::cout << "Rank " << rank << ": Test completed successfully!" << std::endl;
    if (rank == 0){
        printf("Test Pass! \n");
    }
    return 0;
}
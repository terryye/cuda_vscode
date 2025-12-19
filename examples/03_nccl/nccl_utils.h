#ifndef NCCL_UTILS_H
#define NCCL_UTILS_H

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
/**
 * NCCL error checking macro
 */
#define CHECK_NCCL(call) do { \
    ncclResult_t res = call; \
    if (res != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d - %s\n", __FILE__, __LINE__, \
                ncclGetErrorString(res)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}
/**
 * Initialize NCCL communicator with MPI coordination
 * 
 * @param comm Output NCCL communicator handle
 * @param rank MPI rank of current process
 * @param nranks Total number of MPI processes
 * @param mpi_comm MPI communicator (usually MPI_COMM_WORLD)
 * @return 0 on success, -1 on failure
 */
inline int init_nccl_comm(ncclComm_t* comm, int rank, int nranks, MPI_Comm mpi_comm = MPI_COMM_WORLD) {
    ncclUniqueId id;
    
    // Root process generates unique ID
    if (rank == 0) {
        ncclResult_t res = ncclGetUniqueId(&id);
        if (res != ncclSuccess) {
            fprintf(stderr, "Failed to get NCCL unique ID: %s\n", ncclGetErrorString(res));
            return -1;
        }
    }
    
    // Broadcast ID to all processes
    int mpi_ret = MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, mpi_comm);
    if (mpi_ret != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Bcast failed with error code: %d\n", mpi_ret);
        return -1;
    }
    
    // Initialize NCCL communicator
    ncclResult_t res = ncclCommInitRank(comm, nranks, id, rank);
    if (res != ncclSuccess) {
        fprintf(stderr, "Failed to initialize NCCL communicator: %s\n", ncclGetErrorString(res));
        return -1;
    }
    
    return 0;
}

/**
 * Complete initialization: MPI + GPU device + NCCL
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @param rank Output MPI rank
 * @param nranks Output total number of ranks
 * @param comm Output NCCL communicator
 * @return 0 on success, -1 on failure
 */
inline int init_mpi_nccl(int argc, char* argv[], int* rank, int* nranks, ncclComm_t* comm) {
    // Initialize MPI
    int mpi_ret = MPI_Init(&argc, &argv);
    if (mpi_ret != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed\n");
        return -1;
    }
    
    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, nranks);
    
    // Set GPU device (handle multiple ranks per GPU)
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    int device = (*rank) % deviceCount;
    CHECK_CUDA(cudaSetDevice(device));
    
    // Initialize NCCL
    if (init_nccl_comm(comm, *rank, *nranks) != 0) {
        MPI_Finalize();
        return -1;
    }
    
    return 0;
}

/**
 * Cleanup NCCL and MPI resources
 * 
 * @param comm NCCL communicator to destroy
 */
inline void cleanup_mpi_nccl(ncclComm_t comm) {
    CHECK_NCCL(ncclCommDestroy(comm));
    MPI_Finalize();
}
/**
* Perform ring exchange for a single buffer
* 
* @param d_sendbuf Device send buffer pointer
* @param d_recvbuf Device receive buffer pointer
* @param size Size of the buffer to exchange
* @param next_rank Rank of the next process in the ring
* @param prev_rank Rank of the previous process in the ring
* @param comm NCCL communicator
* @param stream CUDA stream for communication
 */
void ring_exchange(float* d_sendbuf, float* d_recvbuf, int size, 
                   int next_rank, int prev_rank, ncclComm_t comm, cudaStream_t stream = 0) {
    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclSend(d_sendbuf, size, ncclFloat, next_rank, comm, stream));
    CHECK_NCCL(ncclRecv(d_recvbuf, size, ncclFloat, prev_rank, comm, stream));
    CHECK_NCCL(ncclGroupEnd());
}

void ring_exchange_multi(float** d_sendbufs, float** d_recvbufs, int size, 
                        int next_rank, int prev_rank, ncclComm_t comm, int num_buffers, cudaStream_t stream = 0) {
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < num_buffers; i++) {
        CHECK_NCCL(ncclSend(d_sendbufs[i], size, ncclFloat, next_rank, comm, stream));
        CHECK_NCCL(ncclRecv(d_recvbufs[i], size, ncclFloat, prev_rank, comm, stream));
    }
    CHECK_NCCL(ncclGroupEnd());
}

void ring_exchange_kv(float* d_sendbuf_k, float* d_recvbuf_k,
                      float* d_sendbuf_v, float* d_recvbuf_v,
                      int size, 
                      int next_rank, int prev_rank, ncclComm_t comm, cudaStream_t stream = 0) {
    float* d_sendbufs[2] = {d_sendbuf_k, d_sendbuf_v};
    float* d_recvbufs[2] = {d_recvbuf_k, d_recvbuf_v};
    ring_exchange_multi(
        d_sendbufs, d_recvbufs,size, next_rank, prev_rank, comm, 2, stream
    );
}

#endif // NCCL_UTILS_H
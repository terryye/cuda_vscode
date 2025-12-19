#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>

// WARNING: To avoid clutter, this code does not do error checking.

__global__ void simple_shift(int* destination) {
    int mype = nvshmem_my_pe();

    // get the number of PEs in the "world" team
    int npes = nvshmem_n_pes();

    // the rank of next peer in the ring
    int peer = (mype + 1) % npes;

    // sets the value of the "destination" symmetric memory
    //  object to the integer value of this rank.
    nvshmem_int_p(destination, mype, peer);
}

int main(int argc, char* argv[]) {
    int rank, nranks;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Initializes nvshmem runtime.
    // Registers the mpi communicator with the nvshmem runtime.
    // Creates a "team" of "procesing elements" (PEs).
    // The "team" corresponds to the MPI communicator group.
    // The "processing elements" corresponds to the MPI processes.
    // Creates a processing element (PE) for each rank.
    nvshmemx_init_attr_t attr;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // get the rank of this PE in the "world" team
    int mype = nvshmem_my_pe();

    // get the number of PEs in the "world" team
    int npes = nvshmem_n_pes();

    // NVSHMEMX_TEAM_NODE is the team of PEs on this node
    // mype_node is the "local" rank of this PE.
    // PEs have a local rank and global rank.
    // The global rank is unique across all PEs across all nodes.
    // The local rank is unique across all PEs only on this.
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    int ndevices;
    cudaGetDeviceCount(&ndevices);

    // Use the "local" rank of this PE to set the device
    // Otherwise, in a multi-node setting, this command__
    // will fail since a global rank would not have a
    // matching device ID.
    cudaSetDevice(mype_node);

    // All operations in the same stream will be done sequentially
    // according to the order in which they are placed into that stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocates a 4-byte symmetric memory object in each rank.
    int* destination = (int*)nvshmem_malloc(sizeof(int));

    // Launch kernel on stream
    simple_shift << <1, 1, 0, stream >> > (destination);

    // This barrier waits for all the PEs to reach this point.
    // The barrier is added to the same stream as the kernel launch.
    //  thus ensuring that ensuring that the kernel completes
    //  before this rank reaches the barrier.
    nvshmemx_barrier_all_on_stream(stream);

    // cudaMemcpyAsync is version of cudaMemcpy that is asynchronous.

    // cudaMemcpy is synchronous in that all cudaMemcpy operations
    // on the host blocks the host before completing, thereby preventing
    // compute from proceeding while the memcpy happens in the background.

    // If the source or destination is on the host, the  cudaMemcpyAsync
    // function expects that the memory is "pinned".
    // "pinned" host memory can be allocated through cudaHostAlloc (instead of malloc).nvshmem
    // "pinned" host memory means that the OS won't page this memory to disk.
    // See chapter 20 in PMPP for more on pinned memory.
    int msg;
    cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // This rank waits until the memcpy async completes.
    cudaStreamSynchronize(stream);

    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    // cleanup both nvshmem
    nvshmem_free(destination);
    nvshmem_finalize();

    // clean up MPI
    MPI_Finalize();

    printf("Test Pass! \n");
    return 0;
}

#!/bin/bash

mkdir -p ./bin
rm -rf ./bin/output.bin
rm -rf ./bin/*.bin.dSYM

# Check that there are 1 or 2 arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 1 ]; then
  echo "Usage: $0 <path_to_cuda_file> "
  exit 1
fi

# Check that $1 is a path to an existing file
if [ ! -f "$1" ]; then
  echo "Error: '$1' is not a valid file."
  exit 2
fi


# Detect number of visible GPUs using nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1; then
  N_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  echo "Number of visible GPUs: $N_GPU"
  # Allow zero matches without exiting under `set -e -o pipefail`
  N_P2P_PAIR=$( { nvidia-smi topo -p2p w | grep "GPU" | grep -o '\bOK\b' || true; } | wc -l )
  echo "Number of P2P capable GPU pairs: $N_P2P_PAIR"

  if [ "$N_P2P_PAIR" -lt "$((N_GPU*(N_GPU-1)/2))" ]; then
    echo -e "\033[33m Alert: Not all GPUs are P2P connected. NVSHMEM may not work as expected. \033[0m"
  fi

  if [ "$N_P2P_PAIR" -eq "0" ]; then
    echo -e "\033[33m Alert: No P2P capable GPU pairs detected. NVSHMEM may not work as expected. \033[0m"
    N_GPU=1
    echo "Setting number of GPUs to 1 for execution."
  fi
else
  echo "nvidia-smi is not found. please install nvidia driver first" >&2
  exit 3
fi

export OMPI_MCA_btl_base_verbose=100 


set -x
# compile and run the code
nvcc -DCUDA=1 -g -G -rdc=true \
       -ccbin \
       mpicxx \
       -lmpi \
       -lstdc++ \
       -lm \
       -lnvidia-ml \
       -lcuda \
       -lnccl \
       -lcudart \
       -lcudadevrt \
       -lnvshmem_host \
       -lnvshmem_device \
        -I/opt/nvshmem/include/ \
        -I/usr/lib/x86_64-linux-gnu/openmpi/include \
        -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
        -L/opt/nvshmem/lib/ \
        -Xlinker -rpath=/opt/nvshmem/lib \
       -o ./bin/output.bin $1

# run the program
echo "this might take a couple of miniutes to start if you are running for the first time..."
mpirun --allow-run-as-root -np ${N_GPU} ./bin/output.bin
#!/bin/bash

set -Eeuoa pipefail
set -x 

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

#Check that if $2 is bound , else default to 1
if [ "$#" -lt 2 ] ; then
    gpu_count=1
else
    gpu_count=1
fi

# Check that $2 is a number (integer or decimal)
if ! [[ "$gpu_count" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: '$gpu_count' is not a valid number."
  exit 3
fi

# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)


CUDA_VERSION=$(nvidia-smi | grep "CUDA" | grep -oP 'CUDA Version: \K[0-9.]+')
CUDA_VER_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_VER_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)


# compile and run the code
nvcc -DCUDA=1 -g -G -rdc=true \
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
        -I${SCRIPT_DIR}/../ \
        -I/usr/lib/x86_64-linux-gnu/openmpi/include \
        -I/usr/include/nvshmem_${CUDA_VER_MAJOR}/ \
        -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
        -L/usr/lib/x86_64-linux-gnu/nvshmem/${CUDA_VER_MAJOR}/ \
       -o ./bin/output.bin $1

# run the program
mpirun --allow-run-as-root \
       -np 2 ./bin/output.bin


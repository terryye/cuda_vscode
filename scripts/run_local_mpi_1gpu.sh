#!/bin/bash

set -Eeuoa pipefail

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

# compile and run the code
nvcc -DCUDA=1 -g -G -rdc=true \
       -arch=sm_75 \
       -lmpi \
       -lstdc++ \
       -lm \
       -lnccl \
       -lcudart \
       -lcudadevrt \
        -I/mnt/c/users/terryye/Documents/Github/neu-hpc-for-ai \
        -I/usr/lib/x86_64-linux-gnu/openmpi/include \
        -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
       -o ./bin/output.bin $1

# run the program
mpirun --allow-run-as-root \
       -np 1 ./bin/output.bin


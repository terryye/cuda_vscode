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

# compile and run the code
nvcc -DCUDA=1 -g -G -rdc=true \
       -arch=sm_75 \
       -lstdc++ \
       -lm \
       -lcudart \
       -lcudadevrt \
        -I/mnt/c/users/terryye/Documents/Github/neu-hpc-for-ai \
        -I/usr/lib/x86_64-linux-gnu/openmpi/include \
        -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
       -o ./bin/output.bin $1

# run the program
./bin/output.bin


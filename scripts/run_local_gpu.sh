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

# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
ARCH="sm_${COMPUTE_CAPABILITY/./}"
# compile and run the code
nvcc -DCUDA=1 -g -G -rdc=true \
       -arch=$ARCH \
       -lstdc++ \
       -lm \
       -lmpi \
       -lcudart \
       -lcudadevrt \
        -I${SCRIPT_DIR}/.. \
        -I/usr/lib/x86_64-linux-gnu/openmpi/include \
        -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
       -o ./bin/output.bin $1

# run the program
./bin/output.bin


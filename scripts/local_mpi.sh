#!/bin/bash

set -Eeuoa pipefail



# Check OS and show alert if Mac or Windows
os=$(uname -s)
if [[ "$os" == "Darwin" ]] || [[ "$os" == *"MINGW"* ]] || [[ "$os" == *"CYGWIN"* ]] || [[ "$os" == "MSYS"* ]]; then
    echo -e "\033[33m Alert: local execution OONLY supported on Linux \033[0m"
    exit 1
fi

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
else
  echo "nvidia-smi is not found. please install nvidia driver first" >&2
  exit 3
fi
# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

set -x
# compile and run the code
nvcc -DCUDA=1 -g -G -rdc=true \
       -lmpi \
       -lstdc++ \
       -lm \
       -lnccl \
       -lcudart \
       -lcudadevrt \
       -I/usr/lib/x86_64-linux-gnu/openmpi/include \
       -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
       -o ./bin/output.bin $1

# run the program
mpirun --allow-run-as-root \
       -np ${N_GPU} ./bin/output.bin


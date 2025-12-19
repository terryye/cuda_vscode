#!/bin/bash
set -eux

### Run this script on an Ubuntu 22 / 24 system with NVIDIA GPU drivers already installed. 

## CUDA toolkit, NCCL, OpenMPI, and NVSHMEM will be installed.
## CUDA 12 / CUDA 13 are supported. 

CUDA_VERSION=$(nvidia-smi | grep "CUDA" | grep -oP 'CUDA Version: \K[0-9.]+')

CUDA_VER_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_VER_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    wget \
    build-essential \
    cmake \
    libhwloc-dev \
    libnuma-dev \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    cuda-toolkit-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} \
    libnvshmem3-cuda-${CUDA_VER_MAJOR} \
    libnvshmem3-dev-cuda-${CUDA_VER_MAJOR}

sudo apt-get install -y libnccl2 libnccl-dev

cat >> ~/.bashrc << 'EOF'
# NVIDIA CUDA Toolkit
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF

source ~/.bashrc
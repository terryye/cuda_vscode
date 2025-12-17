#!/bin/bash
set -eux

CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | cut -d'.' -f1);

# Configuration
export NVSHMEM_HOME = /usr/include/nvshmem_${CUDA_VERSION}

# Install dependencies
apt-get update
apt-get install -y \
    wget \
    build-essential \
    cmake \
    libhwloc-dev \
    libnuma-dev \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libnvshmem3-cuda-${CUDA_VERSION} \
    libnvshmem3-dev-cuda-${CUDA_VERSION}
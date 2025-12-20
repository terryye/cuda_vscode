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
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    build-essential \
    cmake \
    libhwloc-dev \
    libnuma-dev \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libnccl2 \
    libnccl-dev \
    cuda-toolkit-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR} \

NVSHMEM_VERSION="3.4.5-0"
NVSHMEM_PREFIX="/opt/nvshmem"

cd /tmp && \
wget -O nvshmem-${NVSHMEM_VERSION}.tar.gz \
  https://github.com/NVIDIA/nvshmem/archive/refs/tags/v${NVSHMEM_VERSION}.tar.gz && \
tar xvf nvshmem-${NVSHMEM_VERSION}.tar.gz && \
cd nvshmem-${NVSHMEM_VERSION} && \
\
# Disable IBRC/UCX via env so IB (verbs.h) is never needed
export CUDA_HOME=/usr/local/cuda \
  NVSHMEM_IBRC_SUPPORT=0 \
  NVSHMEM_UCX_SUPPORT=0 && \
\
# Configure
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${NVSHMEM_PREFIX} \
  -DNVSHMEM_PREFIX=${NVSHMEM_PREFIX} \
  -DNVSHMEM_MPI_SUPPORT=1 \
  -DNVSHMEM_SHMEM_SUPPORT=0 \
  -DNVSHMEM_UCX_SUPPORT=0 \
  -DNVSHMEM_LIBFABRIC_SUPPORT=0 \
  -DNVSHMEM_IBRC_SUPPORT=0 \
  -DNVSHMEM_BUILD_TESTS=0 \
  -DNVSHMEM_BUILD_EXAMPLES=0 \
  -DNVSHMEM_BUILD_PACKAGES=0 \
  -DNVSHMEM_BUILD_PYTHON_LIB=OFF \
  -DCUDA_ARCHITECTURES=90 && \
\
# Build and install
cmake --build build -j && \
cmake --install build

# Configure NVSHMEM and OpenMPI environment
#export OMPI_ALLOW_RUN_AS_ROOT="1"
#export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1"
#export PMIX_MCA_gds="hash"
#export OMPI_MCA_btl_vader_single_copy_mechanism="none"


cat >> ~/.bashrc << 'EOF'
# NVIDIA CUDA Toolkit
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${NVSHMEM_PREFIX}/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${NVSHMEM_PREFIX}/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

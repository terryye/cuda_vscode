#!/bin/bash
set -eux

CUDA_VERSION=$(nvidia-smi | grep "CUDA" | grep -oP 'CUDA Version: \K[0-9.]+')
CUDA_VER_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_VER_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

# # 1. Download the NVIDIA keyring package
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# # 2. Install the keyring
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# # 3. Update your package list
# sudo apt-get update


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
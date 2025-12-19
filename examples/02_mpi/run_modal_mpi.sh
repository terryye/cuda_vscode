#!/bin/bash
N_GPU=2

modal run ../../scripts/modal_mpi_gpu.py::compile_and_run_cuda_$N_GPU  --code-path ./main.cu
#!/bin/bash

export N_GPU=2
modal run ../../scripts/modal_nvshmem.py  --code-path ./main.cu
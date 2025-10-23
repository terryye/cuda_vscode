#!/bin/bash

set -Eeuao pipefail


mpicc $1 -o output.bin
mpirun -np 4 ./output.bin

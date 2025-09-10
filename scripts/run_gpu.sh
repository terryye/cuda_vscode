#!/bin/bash

set -Eexuoa pipefail

rm -f $1.cu
cp $1 $1.cu
modal run scripts/modal_cu.py --code-path $1.cu
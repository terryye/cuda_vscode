#!/bin/bash

set -Eeuao pipefail

mkdir -p ./bin
rm -rf ./bin/output.bin
rm -rf ./bin/*.bin.dSYM

if [ "$#" -eq 0 ]; then
  echo "need to specify a file run"
  exit 1
fi

# setting this to disable a warning printed when running on MacOS to reduce terminal clutter.
export MallocNanoZone=0

# Set environment variables for AddressSanitizer
export ASAN_OPTIONS="symbolize=1:print_stacktrace=1:halt_on_error=1:print_module_map=0:print_stats=0:print_scariness=0:print_registers=0:color=always:use_color_in_reports=1"
export ASAN_SYMBOLIZER_PATH=/opt/homebrew/opt/llvm/bin/llvm-symbolizer
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

# Try to find llvm-symbolizer
if command -v llvm-symbolizer &> /dev/null; then
    export ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)
elif command -v asan_symbolize &> /dev/null; then
    export ASAN_SYMBOLIZER_PATH=$(which asan_symbolize)
else
    echo "Warning: No symbolizer found. Line numbers may not appear."
fi

clang \
  -Wall -Werror -Wpedantic -Wconversion  -Wsign-conversion  -Wcast-qual \
  -g -O0 \
   -fno-vectorize -fno-slp-vectorize \
  -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope \
  "$1" -o ./bin/output.bin

./bin/output.bin

#!/bin/bash
export CUDA_HOME=/home/chenqi/miniconda3/envs/splinegs
export CC=/home/chenqi/miniconda3/envs/splinegs/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/chenqi/miniconda3/envs/splinegs/bin/x86_64-conda-linux-gnu-c++
export LD_LIBRARY_PATH=/home/chenqi/miniconda3/envs/splinegs/lib/python3.7/site-packages/torch/lib:/home/chenqi/miniconda3/envs/splinegs/lib
export PATH=/home/chenqi/miniconda3/envs/splinegs/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"

exec /home/chenqi/miniconda3/envs/splinegs/bin/python "$@"

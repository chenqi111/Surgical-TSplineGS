#!/bin/bash
# Wrapper script to run evaluation in the surgical_tsplinegs conda environment
cd /home/chenqi/myworker/Surgical-TSplineGS

CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate surgical_tsplinegs 2>/dev/null || true

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$(pwd)/submodules/simple-knn:$PYTHONPATH"

python3 compute_metrics.py 2>&1

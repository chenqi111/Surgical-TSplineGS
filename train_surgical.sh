#!/bin/bash
# Surgical-TSplineGS training script
# Usage: bash train_surgical.sh <scene_path> <exp_name>

SCENE=${1:-"data/surgical/your_scene"}
EXP_NAME=${2:-"surgical_tsplinegs"}

python train.py \
    -s "$SCENE" \
    --expname "$EXP_NAME" \
    --configs arguments/surgical/default.py \
    --dataset_type surgical

#!/bin/bash

# custom config
DATA=/data/
TRAINER=ADAPTER

DEVICE=$1
DATASET_SOURCE=$2      # target dataset - i.e. {imagenet, imagenetv2, imagenet_a, imagenet_r, imagenet_sketch}
DATASET_TARGET=$3      # target dataset - i.e. {imagenetv2, imagenet_a, imagenet_r, imagenet_sketch}
CFG=$4          # config file - SGD_lr1e-1_B256_ep300
SHOTS=$5        # number of shots (1, 2, 4, 8, 16)
INIT=$6         # Method / Linear Probe init - i.e. {RANDOM, ZS, ClipA, TipA, TipA(f), TR, TRenh}
CONSTRAINT=$7   # apply class-adaptive constraint in Linear Probing (CLAP) - i.e. {none, l2}
BACKBONE=$8     # CLIP backbone to sue - i.e. {RN50, RN101, ViT-B/32, ViT-B/16}

for SEED in 1 2 3
do
    MODELDIR=output/FINAL/debug/${DATASET_SOURCE}/${CFG}_${INIT}Init_${CONSTRAINT}Constraint_${SHOTS}shots/seed${SEED}
    OUTDIR=output/FINAL/debug/${DATASET_TARGET}/${CFG}_${INIT}Init_${CONSTRAINT}Constraint_${SHOTS}shots/seed${SEED}
    if [ -d "$OUTDIR" ]; then
        echo "Oops! The results exist at ${OUTDIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET_TARGET}.yaml \
        --config-file configs/trainers/${CFG}.yaml \
        --output-dir ${OUTDIR} \
        --model-dir ${MODELDIR} \
        --load-epoch 300 \
        --eval-only \
        --backbone ${BACKBONE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.ADAPTER.INIT ${INIT} \
        TRAINER.ADAPTER.CONSTRAINT ${CONSTRAINT}
    fi
done
#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1, python tools/run_net.py --cfg configs/movi_e/base.yaml --exp_name steve_x1
# NOTE: remove dependence on the last argument in the parser. put it in the config file.?

# standard format to run files ..

# python tools/run_net.py \
#   --cfg configs/Kinetics/C2D_8x8_R50.yaml \
#   DATA.PATH_TO_DATA_DIR path_to_your_dataset \
#   NUM_GPUS 2 \
#   TRAIN.BATCH_SIZE 16 \
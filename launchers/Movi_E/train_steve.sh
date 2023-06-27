#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate graphOR


#### SETTINGS ####

DATE=`date "+%Y-%m-%d_%H-%M"`
NOTIFY=true
DATA_DIR="/raid/MOVI_E_dataset"
SLOWFAST_CONFIG_PATH="configs/MOVI_E/steve_default_v1.yaml"

# General Training
AVAILABLE_GPUS="0" #,1,2,3" #,3" #,3" #
DISTRIBUTED_PORT=1234 #1234

    # please download pretrain models from the model zoo.

# Backbone Train / Extract
NUM_WORKERS=12 # 8 10
NUM_GPUS=$(( ( ${#AVAILABLE_GPUS} + 1 ) / 2 ))


#### EXECUTION ####

export CUDA_DEVICE_ORDER="PCI_BUS_ID"



CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS \
NCCL_ERROR=DEBUG \
python3 tools/run_net.py --init_method tcp://localhost:${DISTRIBUTED_PORT} \
    --cfg $SLOWFAST_CONFIG_PATH \
    DATA.PATH_TO_DATA_DIR $DATA_DIR \
    NUM_GPUS $NUM_GPUS \
    DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
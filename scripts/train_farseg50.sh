#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='isaid.farseg50'
model_dir='./log/isaid_segm/farseg50'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9996 apex_train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    --opt_level='O1'
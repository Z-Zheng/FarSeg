#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='isaid.farseg50'
model_dir='./log/isaid_segm/farseg50'
ckpt_path='./log/isaid_segm/farseg50/model-60000.pth'
vis_dir='./log/isaid_segm/farseg50/vis-60000'

image_dir='./isaid_segm/val/images'
mask_dir='./isaid_segm/val/masks'

python ./isaid_eval.py \
    --config_path=${config_path} \
    --ckpt_path=${ckpt_path} \
    --image_dir=${image_dir} \
    --mask_dir=${mask_dir} \
    --vis_dir=${vis_dir} \
    --log_dir=${model_dir} \
    --patch_size=896
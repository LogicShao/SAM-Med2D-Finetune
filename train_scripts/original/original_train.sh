#!/bin/bash

# 设置训练参数
WORK_DIR=work_dir
RUN_NAME=sam-med2d-finetune
EPOCHS=20
BATCH_SIZE=2
IMAGE_SIZE=256
DATA_PATH=data_demo
SAM_CHECKPOINT=pretrain_model/sam-med2d_b.pth

# 运行微调脚本
python train.py \
    --work_dir $WORK_DIR \
    --run_name $RUN_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --data_path $DATA_PATH \
    --sam_checkpoint $SAM_CHECKPOINT \
    --encoder_adapter True \
    --use_amp True
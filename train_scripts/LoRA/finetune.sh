#!/bin/bash

# 设置参数
TRAIN_DATA_PATH=dataset_for_SAM-Med2D/train
VAL_DATA_PATH=dataset_for_SAM-Med2D/test
WORK_DIR=brats_finetune_output
RUN_NAME=sam-med2d-brats-lora
MODEL_TYPE=vit_b
SAM_CHECKPOINT=pretrain_model/sam-med2d_b.pth
EPOCHS=100
BATCH_SIZE=4
IMAGE_SIZE=256
LR=1e-5
DEVICE=cuda

python LoRA_finetune/finetune.py \
    --train_data_path $TRAIN_DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --work_dir $WORK_DIR \
    --run_name $RUN_NAME \
    --model_type $MODEL_TYPE \
    --sam_checkpoint $SAM_CHECKPOINT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --lr $LR \
    --device $DEVICE
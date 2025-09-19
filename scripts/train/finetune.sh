python finetune.py \
    --data_path ../Brain-Tumor-Segmentation/BraTs2021/BraTS2021_Training_Data \
    --work_dir ./brats_finetune_output \
    --run_name brats_lora_r8_alpha16 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --sam_checkpoint ./pretrain_model/sam-med2d_b.pth \
    --lora_r 8 \
    --lora_alpha 16

@echo off

python evaluate_finetune.py ^
    --data_path dataset_for_SAM-Med2D/test ^
    --work_dir ./brats_finetune_output ^
    --run_name brats_lora_r8_alpha16 ^
    --sam_checkpoint ./pretrain_model/sam-med2d_b.pth ^
    --metrics dice iou ^
    --subset_size 200

python evaluate_finetune.py ^
    --data_path dataset_for_SAM-Med2D/test ^
    --sam_checkpoint ./pretrain_model/sam-med2d_b.pth ^
    --eval_pretrained_only ^
    --subset_size 200
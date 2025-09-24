@echo off

python LoRA_finetune/evaluate_finetune.py ^
    --data_path dataset_for_SAM-Med2D/test ^
    --work_dir ./brats_finetune_output_1000_200 ^
    --batch_size 128 ^
    --run_name brats_lora_validated ^
    --sam_checkpoint ./pretrain_model/sam-med2d_b.pth ^
    --metrics dice iou ^
    --subset_size 3000

python LoRA_finetune/evaluate_finetune.py ^
    --data_path dataset_for_SAM-Med2D/test ^
    --sam_checkpoint ./pretrain_model/sam-med2d_b.pth ^
    --batch_size 128 ^
    --eval_pretrained_only ^
    --metrics dice iou ^
    --subset_size 3000

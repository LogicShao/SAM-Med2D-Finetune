@echo off

python LoRa_finetune.py ^
    --train_data_path dataset_for_SAM-Med2D/train ^
    --val_data_path dataset_for_SAM-Med2D/test ^
    --work_dir brats_finetune_output ^
    --run_name sam-med2d-brats-lora ^
    --model_type vit_b ^
    --sam_checkpoint pretrain_model\sam-med2d_b.pth ^
    --epochs 5 ^
    --batch_size 64 ^
    --image_size 256 ^
    --lr 1e-5 ^
    --device cuda ^
    --train_subset_size 1000 ^
    --val_subset_size 200

pause

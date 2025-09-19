@echo off

python finetune.py ^
    --train_data_path dataset_for_SAM-Med2D/train ^
    --val_data_path dataset_for_SAM-Med2D/test ^
    --work_dir ./brats_finetune_output ^
    --run_name brats_lora_validated ^
    --lr 1e-5 ^
    --batch_size 64 ^
    --epochs 100 ^
    --early_stopping_patience 10 ^
    --train_subset_size 1000 ^
    --val_subset_size 200
@echo off

set WORK_DIR=work_dir
set RUN_NAME=sam-med2d-finetune
set EPOCHS=200
set BATCH_SIZE=4
set IMAGE_SIZE=256
set DATA_PATH=dataset_for_SAM-Med2D_train
set SAM_CHECKPOINT=pretrain_model\sam-med2d_b.pth

REM 运行微调脚本
python train.py ^
    --work_dir %WORK_DIR% ^
    --run_name %RUN_NAME% ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --image_size %IMAGE_SIZE% ^
    --data_path %DATA_PATH% ^
    --sam_checkpoint %SAM_CHECKPOINT% ^
    --encoder_adapter True ^
    --use_amp True

pause

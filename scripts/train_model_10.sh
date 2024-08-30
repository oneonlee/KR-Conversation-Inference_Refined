#!/bin/bash

# Model 10
number=10
model_path=beomi/Solar-Ko-Recovery-11B
custom_dataset=CustomRefDataset

model_prefix=${model_path#*/}
echo "Model $number Train Start"

python -m run.train \
    --model_id $model_path \
    --custom_dataset $custom_dataset \
    --save_dir "resource/results/$custom_dataset-$model_prefix-LoRA_2e-4" \
    --epoch 4 \
    --lr 2e-4 \
    --gradient_accumulation_steps 64 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj lm_head

echo "Model $number Train Done (Saved at resource/results/$custom_dataset-$model_prefix-LoRA_2e-4)"
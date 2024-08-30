#!/bin/bash

# Model 2
number=2
model_path=x2bee/POLAR-14B-v0.5
custom_dataset=SystemRefOfficialTermDataset

model_prefix=${model_path#*/}
echo "Model $number Train Start"

python -m run.train \
    --model_id $model_path \
    --custom_dataset $custom_dataset \
    --save_dir "resource/results/Refined_$custom_dataset-$model_prefix-QLoRA_2e-4" \
    --train_dataset_path "resource/refined_data/train.json" \
    --valid_dataset_path "resource/refined_data/dev.json" \
    --epoch 4 \
    --lr 2e-4 \
    --gradient_accumulation_steps 64 \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj lm_head

echo "Model $number Train Done (Saved at resource/results/Refined_$custom_dataset-$model_prefix-QLoRA_2e-4)"
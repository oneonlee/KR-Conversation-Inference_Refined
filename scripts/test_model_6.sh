#!/bin/bash

# Model 6
number=6
model_path=x2bee/POLAR-14B-v0.5
custom_dataset=CustomRefDefinitionDataset

model_prefix=${model_path#*/}
echo "Model $number Inference Start"

python -m run.test \
    --load_in_4bit \
    --custom_dataset $custom_dataset \
    --src_model_id $model_path \
    --local_model_path "resource/results/$custom_dataset-$model_prefix-QLoRA_2e-4" \
    --output resource/results/predictions/$custom_dataset-$model_prefix-QLoRA_2e-4.json

echo "Model $number Inference Done (Saved at resource/results/predictions/$custom_dataset-$model_prefix-QLoRA_2e-4.json)"
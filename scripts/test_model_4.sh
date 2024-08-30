#!/bin/bash

# Model 4
number=4
model_path=chihoonlee10/T3Q-ko-solar-dpo-v7.0
custom_dataset=CustomRefInstructionDataset

model_prefix=${model_path#*/}
echo "Model $number Inference Start"

python -m run.test \
    --load_in_4bit \
    --custom_dataset $custom_dataset \
    --src_model_id $model_path \
    --local_model_path "resource/results/$custom_dataset-$model_prefix-LoRA_2e-4" \
    --output resource/results/predictions/$custom_dataset-$model_prefix-LoRA_2e-4.json

echo "Model $number Inference Done (Saved at resource/results/predictions/$custom_dataset-$model_prefix-LoRA_2e-4.json)"
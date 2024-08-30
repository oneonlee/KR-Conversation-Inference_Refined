#!/bin/bash

# Model 9
number=9
model_path=beomi/Solar-Ko-Recovery-11B
custom_dataset=CustomRefDataset

model_prefix=${model_path#*/}
echo "Model $number Inference Start"

python -m run.test \
    --test_dataset_path resource/refined_data/test.json \
    --load_in_4bit \
    --custom_dataset $custom_dataset \
    --src_model_id $model_path \
    --local_model_path "resource/results/Refined_$custom_dataset-$model_prefix-LoRA_2e-4" \
    --output resource/results/predictions/Refined_$custom_dataset-$model_prefix-LoRA_2e-4.json

echo "Model $number Inference Done (Saved at resource/results/predictions/Refined_$custom_dataset-$model_prefix-LoRA_2e-4.json)"
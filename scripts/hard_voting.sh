#!/bin/bash
 
ensemble_recipe="CustomRefOfficialTermDataset-POLAR-14B-v0.2-QLoRA_2e-4 Refined_SystemRefOfficialTermDataset-POLAR-14B-v0.5-QLoRA_2e-4 CustomRefInstructionDataset-Solar-Ko-Recovery-11B-LoRA_1e-4 CustomRefInstructionDataset-T3Q-ko-solar-dpo-v7.0-LoRA_2e-4 CustomRefInstructionDataset-POLAR-14B-v0.5-QLoRA_1e-4 CustomRefDefinitionDataset-POLAR-14B-v0.5-QLoRA_2e-4 SystemRefOfficialTermDataset-Solar-Ko-Recovery-11B-LoRA_1e-4 CustomRefDataset-Solar-Ko-Recovery-11B-LoRA_1e-4 Refined_CustomRefDataset-Solar-Ko-Recovery-11B-LoRA_2e-4 CustomRefDataset-Solar-Ko-Recovery-11B-LoRA_2e-4 CustomRefDefinitionDataset-Solar-Ko-Recovery-11B-LoRA_1e-4"

echo "Start Hard voting Ensemble: $ensemble_recipe"

python -m run.test \
    --ensemble \
    --ensemble_file_list $ensemble_recipe \
    --output resource/results/predictions/final-result.json

echo "End Hard voting Ensemble"
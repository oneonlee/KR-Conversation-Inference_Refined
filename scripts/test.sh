sh scripts/test_model_1.sh
sh scripts/test_model_2.sh
sh scripts/test_model_3.sh
sh scripts/test_model_4.sh
sh scripts/test_model_5.sh
sh scripts/test_model_6.sh
sh scripts/test_model_7.sh
sh scripts/test_model_8.sh
sh scripts/test_model_9.sh
sh scripts/test_model_10.sh
sh scripts/test_model_11.sh

ensemble_file_list="CustomRefOfficialTermDataset-POLAR-14B-v0.2-QLoRA_2e-4 Refined_SystemRefOfficialTermDataset-POLAR-14B-v0.5-QLoRA_2e-4 CustomRefInstructionDataset-Solar-Ko-Recovery-11B-LoRA_1e-4 CustomRefInstructionDataset-T3Q-ko-solar-dpo-v7.0-LoRA_2e-4 CustomRefInstructionDataset-POLAR-14B-v0.5-QLoRA_1e-4 CustomRefDefinitionDataset-POLAR-14B-v0.5-QLoRA_2e-4 SystemRefOfficialTermDataset-Solar-Ko-Recovery-11B-LoRA_1e-4 CustomRefDataset-Solar-Ko-Recovery-11B-LoRA_1e-4 Refined_CustomRefDataset-Solar-Ko-Recovery-11B-LoRA_2e-4 CustomRefDataset-Solar-Ko-Recovery-11B-LoRA_2e-4 CustomRefDefinitionDataset-Solar-Ko-Recovery-11B-LoRA_1e-4"
sh scripts/hard_voting.sh ensemble_file_list
import argparse
import json
import os
import tqdm

import torch
import numpy as np
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.data import CustomRefDataset, CustomRefOfficialTermDataset, CustomRefDefinitionDataset, CustomRefInstructionDataset, SystemRefOfficialTermDataset

parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--test_dataset_path", type=str, default="resource/data/test.json", help="path of test_dataset")
g.add_argument("--submission_file_path", type=str, default="resource/data/test.json", help="path of the submission file")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--src_model_id", type=str, default=None, help="huggingface model id")
g.add_argument("--local_model_path", type=str, default=None, help="local pretrained model path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, default="cuda:0", help="device to load the model")
g.add_argument("--custom_dataset", type=str, default="CustomRefDataset", help="CustomDataset")
g.add_argument("--torch_dtype", type=str, default="bfloat16", help="torch_dtype")
g.add_argument("--load_in_4bit", action="store_true", help="whether use QLoRA")
g.add_argument("--ensemble", action="store_true", help="whether to use ensemble of models")
g.add_argument('--ensemble_file_list', nargs='+', default=[], help='List of file name to ensemble')
g.add_argument("--k_folds", type=int, default=0, help="number of K-folds for ensemble")

def load_model(args, fold=None):
    if args.src_model_id is None:
        if args.load_in_4bit:
            print("`src_model_id` is required!")
        else:
            args.src_model_id = args.local_model_path

    model_path = args.local_model_path
    if fold is not None:
        model_path = os.path.join(args.local_model_path, f"fold_{fold}")

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=args.torch_dtype
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.src_model_id,
            torch_dtype=args.torch_dtype,
            quantization_config=bnb_config,
            device_map=args.device,
        )

        model = PeftModel.from_pretrained(
            model,
            model_path
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device,
        )

    model.eval()
    return model


def load_tokenizer(args, fold=None):
    if args.tokenizer == None:
        args.tokenizer = args.local_model_path

    tokenizer_path = args.tokenizer
    if fold is not None:
        tokenizer_path = os.path.join(args.local_model_path, f"fold_{fold}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return tokenizer


def load_dataset(args, tokenizer):
    """ Load the appropriate dataset based on the custom_dataset argument """
    if args.custom_dataset == "CustomRefDataset":
        return CustomRefDataset(args.test_dataset_path, tokenizer)
    elif args.custom_dataset == "CustomRefOfficialTermDataset":
        return CustomRefOfficialTermDataset(args.test_dataset_path, tokenizer)
    elif args.custom_dataset == "CustomRefDefinitionDataset":
        return CustomRefDefinitionDataset(args.test_dataset_path, tokenizer)
    elif args.custom_dataset == "CustomRefInstructionDataset":
        return CustomRefInstructionDataset(args.test_dataset_path, tokenizer)
    elif args.custom_dataset == "SystemRefOfficialTermDataset":
        return SystemRefOfficialTermDataset(args.test_dataset_path, tokenizer)
    else:
        raise NotImplementedError
    
    
def main(args):
    if args.torch_dtype in ["bf16", "bfloat16"]:
        args.torch_dtype = torch.bfloat16
    elif args.torch_dtype in ["fp16", "float16"]:
        args.torch_dtype = torch.float16
    elif args.torch_dtype in ["fp32", "float32"]:
        args.torch_dtype = torch.float32

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }

    if args.ensemble:
        if args.k_folds > 1:
            # Load tokenizer
            tokenizer = load_tokenizer(args, fold=1)
            tokenizer.pad_token = tokenizer.eos_token

            # Determine dataset type
            dataset = load_dataset(args, tokenizer)

            # Ensemble step 1: Save individual model predictions
            for fold in range(1, args.k_folds + 1):
                fold_output_path = f"{args.output.split('.json')[0]}_fold_{fold}.json"
                if os.path.exists(fold_output_path):
                    continue

                # Load the test dataset
                with open(args.submission_file_path, "r") as f:
                    result = json.load(f)

                model = load_model(args, fold=fold)
                print(f"Processing with model fold_{fold}...")

                # Predict and save each model's results
                for idx in tqdm.tqdm(range(len(dataset)), desc=f"Predicting with fold_{fold}"):
                    inp, _ = dataset[idx]
                    with torch.no_grad():
                        outputs = model(inp.to(args.device).unsqueeze(0))
                        logits = outputs.logits[:, -1, :].flatten()
                        probs = torch.nn.functional.softmax(
                            torch.tensor(
                                [
                                    logits[tokenizer.vocab['A']],
                                    logits[tokenizer.vocab['B']],
                                    logits[tokenizer.vocab['C']],
                                ]
                            ),
                            dim=0,
                        ).detach().cpu().to(torch.float32).numpy()

                    # Save the prediction for this fold
                    result[idx]["output"] = answer_dict[np.argmax(probs)]

                # Save fold predictions to a file
                with open(fold_output_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False, indent=4))

                # Cleanup
                del result
                del model
                torch.cuda.empty_cache()

            # Ensemble step 2: Hard voting across saved predictions
            all_fold_results = []

            for fold in range(1, args.k_folds + 1):
                fold_output_path = f"{args.output.split('.json')[0]}_fold_{fold}.json"
                with open(fold_output_path, "r", encoding="utf-8") as fold_file:
                    all_fold_results.append(json.load(fold_file))

            # Load the test dataset
            with open(args.submission_file_path, "r") as f:
                result = json.load(f)
                
            # Perform hard voting
            final_results = []
            for idx in range(len(result)):
                votes = [fold_results[idx]["output"] for fold_results in all_fold_results]
                # print(f"Votes: {votes}")
                # print(f"vote count: {votes.count}")
                # print(f"Max vote: {max(votes, key=votes.count)}")
                final_results.append(max(votes, key=votes.count))

            # Write final results to the output file
            for idx, final_output in enumerate(final_results):
                result[idx]["output"] = final_output
                
        elif len(args.ensemble_file_list) > 1:
            # Load the test dataset
            with open(args.submission_file_path, "r") as f:
                result = json.load(f)

            all_file_results = []

            for file_name in args.ensemble_file_list:
                if file_name.endswith(".json"):
                    pass
                else:
                    file_name = f"{file_name}.json"
                file_path = os.path.join("resource/results/predictions", file_name)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")


                with open(file_path, "r", encoding="utf-8") as f:
                    file_result = json.load(f)
                    if len(file_result) != len(result):
                        raise ValueError(f"File {file_name} has different length than the submission file")
                    all_file_results.append(file_result)

            # Perform hard voting
            final_results = []
            for idx in range(len(result)):
                votes = [file_results[idx]["output"] for file_results in all_file_results]
                final_results.append(max(votes, key=votes.count))

            # Write final results to the output file
            for idx, final_output in enumerate(final_results):
                result[idx]["output"] = final_output

    else:
        # Load the test dataset
        with open(args.submission_file_path, "r") as f:
            result = json.load(f)

        # Single model inference (no ensemble)
        model = load_model(args)
        tokenizer = load_tokenizer(args)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset(args, tokenizer)

        for idx in tqdm.tqdm(range(len(result)), desc="Inference"):
            inp, _ = dataset[idx]
            with torch.no_grad():
                outputs = model(inp.to(args.device).unsqueeze(0))
                logits = outputs.logits[:, -1, :].flatten()
                probs = torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer.vocab['A']],
                            logits[tokenizer.vocab['B']],
                            logits[tokenizer.vocab['C']],
                        ]
                    ),
                    dim=0,
                ).detach().cpu().to(torch.float32).numpy()

            result[idx]["output"] = answer_dict[np.argmax(probs)]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    exit(main(parser.parse_args()))

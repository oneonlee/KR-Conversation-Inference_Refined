import argparse
import os
import torch
from sklearn.model_selection import KFold
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, LoraConfig, TaskType
from src.data import CustomRefDataset, CustomRefOfficialTermDataset, CustomRefDefinitionDataset, CustomRefInstructionDataset, SystemRefOfficialTermDataset, DataCollatorForSupervisedDataset

parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--custom_dataset", type=str, default="CustomRefDataset", help="CustomDataset")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--save_dir", type=str, default="resource/results", help="model save path")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
g.add_argument("--per_device_train_batch_size", type=int, default=1, help="per_device_train_batch_size")
g.add_argument("--per_device_eval_batch_size", type=int, default=1, help="per_device_eval_batch_size")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
g.add_argument("--epoch", type=int, default=5, help="training epoch")
g.add_argument("--use_all_trainable_data", action="store_true", help="whether use train set or all trainable set (train+eval)")
g.add_argument("--load_in_4bit", action="store_true", help="whether use QLoRA")
g.add_argument("--torch_dtype", type=str, default="bfloat16", help="torch_dtype")
g.add_argument("--full_fine_tuning", action="store_true", help="whether use Full Fine Tuning")
g.add_argument("--lora_r", type=int, default=4, help="LoRA r")
g.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
g.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
g.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "v_proj"], help="LoRA target modules")
g.add_argument("--neftune_noise_alpha", type=int, default=None, help="neftune_noise_alpha")
g.add_argument("--train_dataset_path", type=str, default="resource/data/train.json", help="path of train_dataset")
g.add_argument("--valid_dataset_path", type=str, default="resource/data/dev.json", help="path of valid_dataset")
g.add_argument("--k_folds", type=int, default=5, help="number of K-folds for cross-validation")

def main(args):
    if args.torch_dtype in ["bf16", "bfloat16"]:
        args.torch_dtype = torch.bfloat16
    elif args.torch_dtype in ["fp16", "float16"]:
        args.torch_dtype = torch.float16
    elif args.torch_dtype in ["fp32", "float32"]:
        args.torch_dtype = torch.float32

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=args.torch_dtype
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=args.torch_dtype,
            quantization_config=bnb_config,
            device_map="auto",
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=args.torch_dtype,
            device_map="auto",
        )

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Max Length of `{args.tokenizer}` tokenizer: {tokenizer.model_max_length}")

    # Load the custom dataset
    if args.custom_dataset == "CustomRefDataset":
        train_dataset = CustomRefDataset(args.train_dataset_path, tokenizer, mode="train")
        valid_dataset = CustomRefDataset(args.valid_dataset_path, tokenizer, mode="train")
    elif args.custom_dataset == "CustomRefOfficialTermDataset":
        train_dataset = CustomRefOfficialTermDataset(args.train_dataset_path, tokenizer, mode="train")
        valid_dataset = CustomRefOfficialTermDataset(args.valid_dataset_path, tokenizer, mode="train")
    elif args.custom_dataset == "CustomRefDefinitionDataset":
        train_dataset = CustomRefDefinitionDataset(args.train_dataset_path, tokenizer, mode="train")
        valid_dataset = CustomRefDefinitionDataset(args.valid_dataset_path, tokenizer, mode="train")
    elif args.custom_dataset == "CustomRefInstructionDataset":
        train_dataset = CustomRefInstructionDataset(args.train_dataset_path, tokenizer, mode="train")
        valid_dataset = CustomRefInstructionDataset(args.valid_dataset_path, tokenizer, mode="train")
    elif args.custom_dataset == "SystemRefOfficialTermDataset":
        train_dataset = SystemRefOfficialTermDataset(args.train_dataset_path, tokenizer, mode="train")
        valid_dataset = SystemRefOfficialTermDataset(args.valid_dataset_path, tokenizer, mode="train")
    else:
        raise NotImplementedError

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
    })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
    })

    if args.k_folds > 1:
        kf = KFold(n_splits=args.k_folds)
        fold = 1

        for train_index, val_index in kf.split(train_dataset):
            print(f"Training fold {fold}/{args.k_folds}")
            
            train_fold = train_dataset.select(train_index)
            val_fold = train_dataset.select(val_index)

            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

            # LoRA Config
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
            )

            if not args.full_fine_tuning:
                model = get_peft_model(model, lora_config)

            training_args = SFTConfig(
                output_dir=os.path.join(args.save_dir, f"fold_{fold}"),
                overwrite_output_dir=True,
                do_train=True,
                do_eval=True,
                eval_strategy="epoch",
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.lr,
                weight_decay=0.1,
                num_train_epochs=args.epoch,
                max_steps=-1,
                lr_scheduler_type="cosine",
                warmup_steps=20,
                neftune_noise_alpha=args.neftune_noise_alpha,
                logging_dir="./logs",
                log_level="info",
                logging_steps=1,
                save_strategy="no", # "epoch",
                bf16=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_seq_length=1024,
                packing=True,
                seed=42,
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_fold,
                eval_dataset=val_fold,
                data_collator=data_collator,
                args=training_args,
            )

            trainer.train()
            trainer.save_model()

            # Save the model for this fold
            model.save_pretrained(os.path.join(args.save_dir, f"fold_{fold}"))
            tokenizer.save_pretrained(os.path.join(args.save_dir, f"fold_{fold}"))

            fold += 1

    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        # LoRA Config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )

        if not args.full_fine_tuning:
            model = get_peft_model(model, lora_config)

        training_args = SFTConfig(
            output_dir=args.save_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr,
            weight_decay=0.1,
            num_train_epochs=args.epoch,
            max_steps=-1,
            lr_scheduler_type="cosine",
            warmup_steps=20,
            neftune_noise_alpha=args.neftune_noise_alpha,
            logging_dir="./logs",
            log_level="info",
            logging_steps=1,
            save_strategy="no", # "epoch",
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_seq_length=1024,
            packing=True,
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            args=training_args,
        )

        trainer.train()
        trainer.save_model()

        # Save the model
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

if __name__ == "__main__":
    exit(main(parser.parse_args()))

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mlflow
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


SYSTEM_PROMPT = (
    "You are an expert translator of Akkadian. "
    "Translate the Akkadian transliteration into accurate English. "
    "Return only the English translation."
)


def build_prompt(source: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Akkadian:\n{source}\n\n"
        f"English:\n"
    )


def tokenize_example(example, tokenizer, max_length):
    prompt = build_prompt(example["source_text"])
    target = example["target_text"]

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - len(prompt_ids) - 1,
    )["input_ids"]

    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    input_ids = prompt_ids + target_ids + eos
    labels = [-100] * len(prompt_ids) + target_ids + eos
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def generate_predictions(model, tokenizer, df, max_input_length, max_new_tokens, batch_size):
    model.eval()
    preds = []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        prompts = [build_prompt(x) for x in batch["source_text"]]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Remove the prompt tokens before decoding.
        for i, sequence in enumerate(out):
            prompt_len = int(enc["attention_mask"][i].sum())
            generated = sequence[prompt_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            preds.append(text)

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--output-dir", default="outputs/qwen2.5-7b-lora")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-hf", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA requires a CUDA-capable GPU with bitsandbytes. "
            "This training script is intended for a CUDA machine "
            "(e.g. Colab/Kaggle/cloud GPU), not CPU/MPS."
        )

    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO_ID")

    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing in .env")

    login(token=hf_token)

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=hf_token,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_records = [
        tokenize_example(row, tokenizer, args.max_length)
        for row in train_df.to_dict("records")
    ]
    val_records = [
        tokenize_example(row, tokenizer, args.max_length)
        for row in val_df.to_dict("records")
    ]

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            return self.records[idx]

    train_dataset = ListDataset(train_records)
    val_dataset = ListDataset(val_records)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to=[],
        seed=args.seed,
    )


    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(
        os.getenv("MLFLOW_EXPERIMENT_NAME", "akkadian-qwen2.5-lora")
    )


    with mlflow.start_run() as run:
        mlflow.log_params({
            "base_model": args.model,
            "architecture": "qwen2.5-7b-qlora",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "epochs": args.epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed,
            "train_size": len(train_df),
            "val_size": len(val_df),
        })

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )

        train_result = trainer.train()
        eval_metrics = trainer.evaluate()

        train_metrics = train_result.metrics
        if "train_loss" in train_metrics:
            mlflow.log_metric("train_loss", float(train_metrics["train_loss"]))
        if "eval_loss" in eval_metrics:
            mlflow.log_metric("val_loss", float(eval_metrics["eval_loss"]))

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Validation generation and MT metrics.
        preds = generate_predictions(
            model,
            tokenizer,
            val_df,
            args.max_length,
            args.max_new_tokens,
            args.eval_batch_size,
        )

        references = val_df["target_text"].tolist()
        bleu = BLEU().corpus_score(preds, [references]).score
        chrfpp = CHRF(word_order=2).corpus_score(preds, [references]).score

        mlflow.log_metric("val_BLEU", float(bleu))
        mlflow.log_metric("val_chrF++", float(chrfpp))

        pred_df = val_df.copy()
        pred_df["pred_translation"] = preds
        pred_path = output_dir / "preds_val_qwen2.5-7b-lora.csv"
        pred_df.rename(columns={"oare_id": "id"}, inplace=True)
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")

        if hf_repo and not args.skip_hf:
            # Push only the LoRA adapter + tokenizer, not the 7B base model.
            model.push_to_hub(
                hf_repo,
                private=True,
                token=hf_token,
                commit_message="Upload Akkadian QLoRA adapter",
            )
            tokenizer.push_to_hub(
                hf_repo,
                private=True,
                token=hf_token,
                commit_message="Upload tokenizer",
            )

            mlflow.set_tag("hf_repo", hf_repo)

        mlflow.set_tag("model_architecture", "qwen2.5-lora")
        mlflow.log_dict(
            {
                "run_id": run.info.run_id,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "val_BLEU": bleu,
                "val_chrF++": chrfpp,
            },
            "run_summary.json",
        )

        print(json.dumps({
            "run_id": run.info.run_id,
            "train_loss": train_metrics.get("train_loss"),
            "val_loss": eval_metrics.get("eval_loss"),
            "val_BLEU": bleu,
            "val_chrF++": chrfpp,
            "hf_repo": hf_repo,
        }, indent=2))


if __name__ == "__main__":
    main()

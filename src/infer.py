import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "You are an expert translator of Akkadian. "
    "Translate the Akkadian transliteration into accurate English. "
    "Return only the English translation."
)


def build_prompt(source: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nAkkadian:\n{source}\n\nEnglish:\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing")

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA inference in this script requires CUDA.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token,
    )
    model = PeftModel.from_pretrained(base, args.adapter, token=token)
    model.eval()

    df = pd.read_csv(args.input)
    if "source_text" not in df.columns:
        if "transliteration_clean" in df.columns:
            df["source_text"] = df["transliteration_clean"]
        else:
            raise ValueError("Input needs source_text or transliteration_clean")

    predictions = []
    for source in df["source_text"].astype(str):
        prompt = build_prompt(source)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = out[0][enc["input_ids"].shape[1]:]
        predictions.append(
            tokenizer.decode(generated, skip_special_tokens=True).strip()
        )

    result = pd.DataFrame({
        "id": df["oare_id"] if "oare_id" in df.columns else df["id"],
        "source_text": df["source_text"],
        "target_text": (
            df["target_text"] if "target_text" in df.columns else ""
        ),
        "pred_translation": predictions,
    })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()

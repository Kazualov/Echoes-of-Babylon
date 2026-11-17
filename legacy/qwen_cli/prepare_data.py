import argparse
import hashlib
from pathlib import Path

import pandas as pd


def stable_bucket(value: str, seed: int = 42) -> int:
    h = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/train_clean.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = [
        "oare_id",
        "transliteration_clean",
        "translation",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()
    df = df.dropna()
    df["source_text"] = df["transliteration_clean"].astype(str).str.strip()
    df["target_text"] = df["translation"].astype(str).str.strip()

    df = df[
        (df["source_text"] != "")
        & (df["target_text"] != "")
    ].copy()

    if df["oare_id"].duplicated().any():
        raise ValueError("Duplicate oare_id detected.")

    # Stable split: all team members can reproduce exactly the same validation set.
    df["_bucket"] = df["oare_id"].map(
        lambda x: stable_bucket(str(x), args.seed)
    )

    val_cutoff = args.val_percent
    val_df = df[df["_bucket"] < val_cutoff].copy()
    train_df = df[df["_bucket"] >= val_cutoff].copy()

    train_df = train_df.drop(columns=["_bucket"])
    val_df = val_df.drop(columns=["_bucket"])

    # Keep the canonical columns used by all models / ensemble code.
    columns = ["oare_id", "source_text", "target_text"]
    train_df = train_df[columns].sort_values("oare_id").reset_index(drop=True)
    val_df = val_df[columns].sort_values("oare_id").reset_index(drop=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)

    print(f"Total: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Train/val ratio: {len(train_df)/len(df):.3f}/{len(val_df)/len(df):.3f}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()

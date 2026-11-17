"""Shared document-level train/validation split for every notebook in this project.

All models (mBART, ByT5, Qwen, the ensemble) are validated on exactly the same
documents. The split is deterministic: sorted unique `oare_id`s are shuffled
with random.Random(SPLIT_SEED) and the first int(n * VAL_FRACTION) become
validation. `make_split` checks the result against EXPECTED_VAL_SHA256 so that a
notebook stops if its split ever differs from the reference one.

The same source is pasted into each notebook by `tools/sync_shared_cells.py`;
edit this file, then re-run that script. Do not edit the notebook copies.
"""
import hashlib
import random

ID_COL = "oare_id"
VAL_FRACTION = 0.1
SPLIT_SEED = 42
# sha256 of "\n".join(sorted(val_ids)) for the competition train.csv (1561 documents -> 156 val).
EXPECTED_VAL_SHA256 = "1939161cfcb9ff85082f20b36be2362da429f1da14f8a94f289249ffb78e1304"
EXPECTED_N_VAL = 156


def val_ids_sha256(val_ids):
    return hashlib.sha256("\n".join(sorted(map(str, val_ids))).encode("utf-8")).hexdigest()


def make_split(df, id_col=ID_COL, val_fraction=VAL_FRACTION, seed=SPLIT_SEED, check=True):
    """Return (train_df, val_df), split by document id. Rows with empty text are dropped first."""
    df = df.dropna(subset=["transliteration", "translation"]).reset_index(drop=True)
    doc_ids = sorted(df[id_col].astype(str).unique().tolist())
    random.Random(seed).shuffle(doc_ids)
    n_val = max(1, int(len(doc_ids) * val_fraction))
    val_ids = set(doc_ids[:n_val])
    is_val = df[id_col].astype(str).isin(val_ids)
    train_df = df[~is_val].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)
    digest = val_ids_sha256(val_ids)
    print(f"[shared split] train={len(train_df)} rows, val={len(val_df)} rows "
          f"({n_val} documents), val_ids_sha256={digest[:16]}...")
    if check:
        assert n_val == EXPECTED_N_VAL and digest == EXPECTED_VAL_SHA256, (
            "Validation split differs from the shared reference split "
            f"(n_val={n_val}, sha256={digest}). Is train.csv the original competition file?")
    assert not set(train_df[id_col].astype(str)) & val_ids, "document leakage between train and val"
    return train_df, val_df

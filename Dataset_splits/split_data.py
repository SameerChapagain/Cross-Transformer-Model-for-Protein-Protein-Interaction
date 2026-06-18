"""
split_data.py
─────────────
Reads protein.pairs_9606.tsv (no header, 3 cols: protein1  protein2  label)
and writes stratified train / val / test splits.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── CONFIG ──────────────────────────────────────────────
PAIRS_FILE = "protein.pairs_9606.tsv"
OUTPUT_DIR = "splits"
TEST_SIZE  = 0.10
VAL_SIZE   = 0.10
SEED       = 42\

# ────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = pd.read_csv(
        PAIRS_FILE, sep="\t", header=None,
        names=["protein1", "protein2", "label"]
    )
    print(f"Total pairs : {len(data)}")
    print(f"  Positives : {int(data['label'].sum())}")
    print(f"  Negatives : {int((data['label'] == 0).sum())}")

    # first split → train+val  vs  test
    train_val, test = train_test_split(
        data, test_size=TEST_SIZE,
        random_state=SEED, stratify=data["label"]
    )
    # second split → train  vs  val
    relative_val = VAL_SIZE / (1.0 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=relative_val,
        random_state=SEED, stratify=train_val["label"]
    )

    train.to_csv(f"{OUTPUT_DIR}/train.tsv", sep="\t", index=False, header=False)
    val.to_csv(  f"{OUTPUT_DIR}/val.tsv",   sep="\t", index=False, header=False)
    test.to_csv( f"{OUTPUT_DIR}/test.tsv",  sep="\t", index=False, header=False)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  Train : {len(train)}")
    print(f"  Val   : {len(val)}")
    print(f"  Test  : {len(test)}")

if __name__ == "__main__":
    main()
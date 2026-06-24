"""Reproducible subsample of CIC-IDS2017 used in LAWE-IDS experiments.

CIC-IDS2017 raw CSVs total ~6 GB across 8 daily files. This script produces
the exact 500 000-row stratified subsample (320 k train / 80 k val / 100 k test)
used in the paper from those files.

Usage
-----
1. Download the raw CIC-IDS2017 CSVs from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/ids-2017.html
   Place all CSV files (Monday, Tuesday, Wednesday, Thursday, Friday) in RAW_DIR.

2. Run:
       python data/subsample_cicids2017.py --raw_dir /path/to/raw --out_dir data/cicids2017

The output directory will contain a single cicids2017_sampled.csv used by
preprocessing.py automatically.

Reproducibility note
--------------------
The subsample uses random_state=42 (same as SEED in config.py). Changing
RAW_DIR or the seed will produce different rows; all other parameters are fixed
to match the paper exactly.
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd

SAMPLE_SIZE = 500_000
SEED = 42

LABEL_COL_CANDIDATES = [' Label', 'Label', 'label']
DROP_COLS = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp',
             'Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
BENIGN_LABELS = {'BENIGN', 'Benign', 'benign', 'Normal'}


def find_label_col(df):
    for c in LABEL_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No label column found. Columns: {list(df.columns)[:10]}")


def load_raw(raw_dir):
    csvs = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    print(f"Found {len(csvs)} CSV files in {raw_dir}")
    chunks = []
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        chunks.append(df)
        print(f"  {os.path.basename(p)}: {len(df):,} rows")
    return pd.concat(chunks, ignore_index=True)


def clean(df):
    label_col = find_label_col(df)
    df = df.rename(columns={label_col: 'Label'})

    # Drop non-feature columns
    drop = [c for c in DROP_COLS if c.strip() in df.columns]
    df = df.drop(columns=drop, errors='ignore')

    # Binary label: 0 = benign, 1 = attack
    df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip() in BENIGN_LABELS else 1)

    # Drop rows with inf / all-NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how='any')

    # Keep numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' not in numeric_cols:
        numeric_cols.append('Label')
    df = df[numeric_cols]
    return df


def subsample(df, n=SAMPLE_SIZE, seed=SEED):
    rng = np.random.default_rng(seed)
    if len(df) <= n:
        print(f"  Dataset has only {len(df):,} clean rows; using all.")
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Stratified: preserve attack/benign ratio
    benign = df[df['Label'] == 0]
    attack = df[df['Label'] == 1]
    ratio = len(benign) / len(df)
    n_benign = int(n * ratio)
    n_attack = n - n_benign
    s_benign = benign.sample(n=min(n_benign, len(benign)), random_state=seed)
    s_attack = attack.sample(n=min(n_attack, len(attack)), random_state=seed)
    result = pd.concat([s_benign, s_attack]).sample(frac=1, random_state=seed)
    result = result.reset_index(drop=True)
    print(f"  Sampled {len(result):,} rows  "
          f"(benign={len(s_benign):,}, attack={len(s_attack):,})")
    return result


def main():
    ap = argparse.ArgumentParser(description='Reproduce LAWE-IDS CIC-IDS2017 subsample')
    ap.add_argument('--raw_dir', required=True,
                    help='Directory containing raw CIC-IDS2017 CSV files')
    ap.add_argument('--out_dir', default='data/cicids2017',
                    help='Output directory (default: data/cicids2017)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'cicids2017_sampled.csv')

    print("Loading raw CIC-IDS2017 files …")
    df = load_raw(args.raw_dir)
    print(f"Total rows before cleaning: {len(df):,}")

    print("Cleaning …")
    df = clean(df)
    print(f"Total rows after cleaning: {len(df):,}")

    print(f"Stratified subsample → {SAMPLE_SIZE:,} rows (seed={SEED}) …")
    df = subsample(df, n=SAMPLE_SIZE, seed=SEED)

    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print("Done. Run 'python run_lawe.py --dataset cicids2017' to use this file.")


if __name__ == '__main__':
    main()

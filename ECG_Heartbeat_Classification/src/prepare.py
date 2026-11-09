"""Task 1 - Load and prepare the data.

Run from the repository root:   python src/prepare.py
Other scripts import prepare() to get the ready-made tensors.
"""
import json
import os

import numpy as np
import torch

from load_data import load_raw

CLASS_LETTERS = ["N", "S", "V", "F", "Q"]
N_FEATURES = 187
LABEL_COL = 187

# Row counts given in the assignment (Section 1) - used for the check in Task 1.4
EXPECTED_TRAIN = [72471, 2223, 5788, 641, 6431]
EXPECTED_TEST = [18118, 556, 1448, 162, 1608]


def _split_xy(df, name, verbose):
    """Task 1.2: separate the signal from the label and make labels integers."""
    X = df.iloc[:, :N_FEATURES].to_numpy()          # columns 0..186
    labels_raw = df.iloc[:, LABEL_COL].to_numpy()   # column 187
    if verbose:
        print(f"[{name}] signal shape = {X.shape}, label column dtype = {labels_raw.dtype}")
    # The label column is stored as float (0.0, 1.0, ...). Check they are whole numbers.
    if not np.all(labels_raw == np.round(labels_raw)):
        raise ValueError(f"{name}: label column contains non-integer values")
    y = labels_raw.astype(np.int64)
    if y.min() < 0 or y.max() > 4:
        raise ValueError(f"{name}: labels outside 0..4 found")
    if verbose:
        print(f"[{name}] labels converted to {y.dtype}, unique values = {np.unique(y).tolist()}")
    return X, y, str(labels_raw.dtype)


def prepare(data_dir="data", verbose=True):
    """Return (data, info). `data` holds the six tensors, `info` the numbers for Table 1."""
    raw = load_raw(data_dir)
    train_df, test_df = raw["train"], raw["test"]

    # ---- Task 1.1: shapes
    if verbose:
        print(f"train file shape: {train_df.shape}")
        print(f"test  file shape: {test_df.shape}")

    # ---- Task 1.2: signal / label
    X_all, y_all, label_dtype = _split_xy(train_df, "train", verbose)
    X_test, y_test, _ = _split_xy(test_df, "test", verbose)
    assert X_all.shape[1] == N_FEATURES, "input must have 187 columns (label not included)"

    # ---- Task 1.3: min / max of the signal in the training file (used as is)
    sig_min, sig_max = float(X_all.min()), float(X_all.max())
    if verbose:
        print(f"training signal: min = {sig_min}, max = {sig_max}")

    # ---- Task 1.4: rows per class, checked against the table in the assignment
    train_counts = np.bincount(y_all, minlength=5).tolist()
    test_counts = np.bincount(y_test, minlength=5).tolist()
    if verbose:
        print("rows per class (train file):", dict(zip(CLASS_LETTERS, train_counts)))
        print("rows per class (test  file):", dict(zip(CLASS_LETTERS, test_counts)))
    if train_counts != EXPECTED_TRAIN or test_counts != EXPECTED_TEST:
        raise ValueError(
            "Class counts do not match the table in the assignment.\n"
            f"  train: got {train_counts}, expected {EXPECTED_TRAIN}\n"
            f"  test : got {test_counts}, expected {EXPECTED_TEST}"
        )
    if verbose:
        print("class counts match the assignment table: OK")

    # ---- Task 1.5 (split): exactly the code given in the assignment
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    if verbose:
        print(f"rows: train = {len(train_idx)}, validation = {len(val_idx)}, test = {len(y_test)}")
        print("validation rows per class:", dict(zip(CLASS_LETTERS, np.bincount(y_val, minlength=5).tolist())))

    # ---- Task 1.6: tensors (float32 signals, long labels)
    data = {
        "X_train": torch.from_numpy(X_train).to(torch.float32),
        "y_train": torch.from_numpy(y_train).to(torch.long),
        "X_val": torch.from_numpy(X_val).to(torch.float32),
        "y_val": torch.from_numpy(y_val).to(torch.long),
        "X_test": torch.from_numpy(X_test).to(torch.float32),
        "y_test": torch.from_numpy(y_test).to(torch.long),
    }
    if verbose:
        for k, v in data.items():
            print(f"{k:8s} shape = {tuple(v.shape)}, dtype = {v.dtype}")

    info = {
        "train_file_shape": list(train_df.shape),
        "test_file_shape": list(test_df.shape),
        "label_dtype_in_csv": label_dtype,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(y_test)),
        "signal_min": sig_min,
        "signal_max": sig_max,
        "train_file_counts": train_counts,
        "test_file_counts": test_counts,
    }
    return data, info


def write_table1(info, path="results/table1_data.md"):
    fmt = lambda c: ", ".join(f"{l} = {n:,}" for l, n in zip(CLASS_LETTERS, c))
    text = (
        "# Table 1 - Data\n\n"
        "| Item | Value |\n|---|---|\n"
        f"| Train / validation / test rows | {info['train_rows']:,} / {info['val_rows']:,} / {info['test_rows']:,} |\n"
        f"| Minimum / maximum signal value (train file) | {info['signal_min']:.4f} / {info['signal_max']:.4f} |\n"
        f"| Rows per class, train file | {fmt(info['train_file_counts'])} |\n"
        f"| Rows per class, test file | {fmt(info['test_file_counts'])} |\n"
    )
    with open(path, "w") as f:
        f.write(text)
    print(f"saved {path}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    _, info = prepare()
    write_table1(info)
    with open("results/data_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("saved results/data_info.json")

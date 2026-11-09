"""Task 5 - Test once.

Run from the repository root:   python src/test_model.py

Loads the saved parameters (best validation epoch of the chosen alpha) and predicts
every test row (class with the largest logit). The test rows are used only here.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics import accuracy, balanced_accuracy, confusion_matrix, per_class_accuracy
from model import make_model
from prepare import CLASS_LETTERS, EXPECTED_TEST, prepare

MODEL_PATH = "results/ecgnet_best.pt"


def main():
    os.makedirs("results", exist_ok=True)
    data, _ = prepare(verbose=False)
    X_test, y_test = data["X_test"], data["y_test"].numpy()

    model = make_model(0)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
    y_pred = logits.argmax(dim=1).numpy()          # class with the largest logit

    C = confusion_matrix(y_test, y_pred)
    acc = accuracy(y_test, y_pred)
    pca = per_class_accuracy(C)
    bal = balanced_accuracy(C)

    # always-"N" baseline computed from the class table
    always_n = EXPECTED_TEST[0] / sum(EXPECTED_TEST)

    # ---- correctness checks from the assignment
    assert C.sum() == 21892, "confusion matrix must add up to 21,892"
    assert C.sum(axis=1).tolist() == EXPECTED_TEST, "each row must add up to that class's test count"

    print(f"test accuracy      : {acc * 100:.2f}%")
    print(f"balanced accuracy  : {bal * 100:.2f}%")
    print(f"always-N accuracy  : {always_n * 100:.2f}%")
    for l, a in zip(CLASS_LETTERS, pca):
        print(f"per-class accuracy {l}: {a * 100:.2f}%")
    print("confusion matrix (rows = true, columns = predicted):")
    print(C)

    # most frequent wrong prediction of the weakest class
    weakest = int(np.argmin(pca))
    off = C[weakest].copy()
    off[weakest] = 0
    confused_with = int(np.argmax(off))
    print(f"lowest per-class accuracy: {CLASS_LETTERS[weakest]}; most often mistaken for "
          f"{CLASS_LETTERS[confused_with]} ({off[confused_with]} of {C[weakest].sum()} rows)")

    # ---- Table 4
    L = CLASS_LETTERS
    t4 = ["# Table 4 - Test results\n",
          "| | " + " | ".join(L) + " |", "|---|" + "---|" * 5,
          "| Per-class accuracy | " + " | ".join(f"{a * 100:.2f}%" for a in pca) + " |\n",
          "| Metric | Value |", "|---|---|",
          f"| Test accuracy | {acc * 100:.2f}% |",
          f"| Balanced accuracy | {bal * 100:.2f}% |",
          f"| Always-\"N\" accuracy (Task 6.1) | {always_n * 100:.2f}% |\n",
          "## Confusion matrix (rows = true class, columns = predicted class)\n",
          "| true \\ predicted | " + " | ".join(L) + " | row total |", "|---|" + "---|" * 6]
    for j in range(5):
        t4.append(f"| **{L[j]}** | " + " | ".join(str(v) for v in C[j]) + f" | {C[j].sum()} |")
    with open("results/table4_test_results.md", "w") as f:
        f.write("\n".join(t4) + "\n")

    # ---- heat-map
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(L); ax.set_yticklabels(L)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title("Test confusion matrix")
    for j in range(5):
        for k in range(5):
            ax.text(k, j, f"{C[j, k]:,}", ha="center", va="center",
                    color="white" if C[j, k] > C.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    plt.close(fig)

    with open("results/test_results.json", "w") as f:
        json.dump({"test_accuracy": acc, "balanced_accuracy": bal, "always_N_accuracy": always_n,
                   "per_class_accuracy": dict(zip(L, pca.tolist())),
                   "confusion_matrix": C.tolist(),
                   "weakest_class": L[weakest], "most_confused_with": L[confused_with]}, f, indent=2)
    print("saved results/table4_test_results.md, confusion_matrix.png, test_results.json")


if __name__ == "__main__":
    main()

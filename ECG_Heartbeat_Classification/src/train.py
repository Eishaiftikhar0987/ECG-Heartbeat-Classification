"""Task 4 - Train.

Run from the repository root:   python src/train.py

For every parameter P a velocity v_P (starting at zero) is kept and, after
loss.backward(), inside torch.no_grad():

    g   = dJ/dP + lambda * P        (weights only; for biases g = dJ/dP)
    v_P = beta * v_P + (1 - beta) * g
    P   = P - alpha * v_P

No torch.optim is used. The test rows are never touched in this file.
"""
import argparse
import json
import math
import os
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import cross_entropy, count_parameters, make_model
from prepare import prepare

# ---- settings fixed by the assignment
BATCH_SIZE = 128
BETA = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS = 30
ALPHAS = [0.01, 0.1, 0.5]
SEED = 0


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_name(device):
    if device.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return f"CPU: {platform.processor() or platform.machine()}"


@torch.no_grad()
def evaluate(model, X, y, chunk=20000):
    """Mean cross-entropy and accuracy of the model on (X, y)."""
    total_loss, correct = 0.0, 0
    for i in range(0, len(X), chunk):
        xb, yb = X[i:i + chunk], y[i:i + chunk]
        logits = model(xb)
        total_loss += cross_entropy(logits, yb).item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def train(alpha, data, device, epochs=EPOCHS, verbose=True):
    """Train a fresh network (seed 0) and return its history and best-epoch weights."""
    model = make_model(SEED).to(device)            # same initial weights on CPU and GPU
    params = list(model.parameters())              # W1, b1, W2, b2, W3, b3 ...
    is_weight = [p.ndim == 2 for p in params]      # weight decay on weight matrices only
    velocity = [torch.zeros_like(p) for p in params]

    X_tr, y_tr = data["X_train"].to(device), data["y_train"].to(device)
    X_va, y_va = data["X_val"].to(device), data["y_val"].to(device)
    n = len(X_tr)

    gen = torch.Generator()
    gen.manual_seed(SEED)                          # reproducible shuffling

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = {"val_loss": math.inf, "epoch": 0, "state": None}
    diverged = False

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen).to(device)     # shuffle every epoch
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            loss = cross_entropy(model(X_tr[idx]), y_tr[idx])

            for p in params:
                p.grad = None                                  # clear old gradients
            loss.backward()

            with torch.no_grad():
                for p, v, w in zip(params, velocity, is_weight):
                    g = p.grad + WEIGHT_DECAY * p if w else p.grad
                    v.mul_(BETA).add_((1.0 - BETA) * g)        # v = beta*v + (1-beta)*g
                    p.sub_(alpha * v)                          # P = P - alpha*v

        tr_loss, tr_acc = evaluate(model, X_tr, y_tr)
        va_loss, va_acc = evaluate(model, X_va, y_va)
        for k, val in zip(hist, (tr_loss, va_loss, tr_acc, va_acc)):
            hist[k].append(val)

        if verbose:
            print(f"  alpha={alpha:<5} epoch {epoch:2d}/{epochs}  "
                  f"train loss {tr_loss:.4f} acc {tr_acc * 100:6.2f}%  |  "
                  f"val loss {va_loss:.4f} acc {va_acc * 100:6.2f}%")

        if not math.isfinite(va_loss):                         # training blew up (e.g. alpha too large)
            diverged = True
            for _ in range(epoch, epochs):                     # keep all histories 30 long
                for k in hist:
                    hist[k].append(float("nan"))
            if verbose:
                print(f"  alpha={alpha}: loss is not finite - training diverged, stopped early")
            break

        if va_loss < best["val_loss"]:
            best.update(val_loss=va_loss, epoch=epoch,
                        state={k: t.detach().cpu().clone() for k, t in model.state_dict().items()})

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0

    return {"alpha": alpha, "history": hist, "best": best, "seconds": seconds, "diverged": diverged}


# ----------------------------------------------------------------------------- plots / tables
def plot_lr_comparison(runs, path):
    plt.figure(figsize=(7, 4.5))
    for r in runs:
        plt.plot(range(1, len(r["history"]["val_loss"]) + 1), r["history"]["val_loss"],
                 marker="o", ms=3, label=f"alpha = {r['alpha']}")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("validation loss (log scale)")
    plt.title("Validation loss for the three learning rates")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_chosen(run, loss_path, acc_path):
    h, best_ep = run["history"], run["best"]["epoch"]
    ep = range(1, len(h["train_loss"]) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, h["train_loss"], label="training loss")
    plt.plot(ep, h["val_loss"], label="validation loss")
    plt.axvline(best_ep, color="gray", ls="--", lw=1, label=f"best epoch ({best_ep})")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(f"Training and validation loss (alpha = {run['alpha']})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, [a * 100 for a in h["train_acc"]], label="training accuracy")
    plt.plot(ep, [a * 100 for a in h["val_acc"]], label="validation accuracy")
    plt.axvline(best_ep, color="gray", ls="--", lw=1, label=f"best epoch ({best_ep})")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.title(f"Training and validation accuracy (alpha = {run['alpha']})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()


def best_row(run):
    """Validation loss and accuracy etc. at the epoch with the lowest validation loss."""
    e = run["best"]["epoch"]
    if e == 0:
        return None
    h = run["history"]
    return {k: h[k][e - 1] for k in h}


def write_table3(runs, path):
    lines = ["# Table 3 - Learning-rate choice\n",
             "| alpha | Lowest validation loss | Validation accuracy at that epoch | Epoch |",
             "|---|---|---|---|"]
    for r in runs:
        row = best_row(r)
        if row is None:
            lines.append(f"| {r['alpha']} | diverged (nan) | - | - |")
        else:
            lines.append(f"| {r['alpha']} | {row['val_loss']:.4f} | {row['val_acc'] * 100:.2f}% | {r['best']['epoch']} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_table2(chosen, init_loss, n_params, device, path):
    row = best_row(chosen)
    gc = None
    if os.path.exists("results/grad_check.json"):
        with open("results/grad_check.json") as f:
            gc = json.load(f)
    gc_text = (f"{gc['max_abs_diff_W3']:.2e} / {gc['max_abs_diff_b3']:.2e}" if gc
               else "run `python src/grad_check.py` first")
    text = (
        "# Table 2 - Network and training\n\n"
        "| Item | Value |\n|---|---|\n"
        f"| Trainable parameters | {n_params:,} |\n"
        f"| Initial loss (before training, whole training set) | {init_loss:.4f}  (log 5 = {math.log(5):.4f}) |\n"
        f"| Gradient check: max difference for W[3] / b[3] | {gc_text} |\n"
        f"| Chosen alpha | {chosen['alpha']} |\n"
        f"| Best epoch (lowest validation loss) | {chosen['best']['epoch']} |\n"
        f"| Training / validation loss at best epoch | {row['train_loss']:.4f} / {row['val_loss']:.4f} |\n"
        f"| Training / validation accuracy at best epoch | {row['train_acc'] * 100:.2f}% / {row['val_acc'] * 100:.2f}% |\n"
        f"| Training time and hardware | {chosen['seconds']:.1f} s for 30 epochs on {hardware_name(device)} |\n"
    )
    with open(path, "w") as f:
        f.write(text)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="keep 30 for the assignment")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    device = get_device()
    print(f"device: {hardware_name(device)}")

    data, _ = prepare(args.data_dir, verbose=False)
    print(f"train {tuple(data['X_train'].shape)}, validation {tuple(data['X_val'].shape)}")

    # Initial loss of the freshly initialised network (should be close to log 5 = 1.61)
    init_model = make_model(SEED).to(device)
    n_params = count_parameters(init_model)
    init_loss, _ = evaluate(init_model, data["X_train"].to(device), data["y_train"].to(device))
    print(f"trainable parameters: {n_params:,}")
    print(f"initial training loss: {init_loss:.4f}   (log 5 = {math.log(5):.4f})")

    # ---- Task 4.2: run the three learning rates
    runs = []
    for alpha in ALPHAS:
        print(f"\n=== training with alpha = {alpha} ===")
        runs.append(train(alpha, data, device, epochs=args.epochs))

    def score(r):                                    # diverged runs can never be chosen
        return r["best"]["val_loss"] if r["best"]["state"] is not None else math.inf

    chosen = min(runs, key=score)
    print(f"\nchosen alpha = {chosen['alpha']} (lowest validation loss {chosen['best']['val_loss']:.4f} "
          f"at epoch {chosen['best']['epoch']})")

    # ---- figures
    plot_lr_comparison(runs, "results/lr_val_loss.png")
    plot_chosen(chosen, "results/loss_curves.png", "results/accuracy_curves.png")

    # ---- Task 4.3: save the weights of the epoch with the lowest validation loss
    torch.save(chosen["best"]["state"], "results/ecgnet_best.pt")
    with open("results/best_model_info.json", "w") as f:
        json.dump({"alpha": chosen["alpha"], "best_epoch": chosen["best"]["epoch"],
                   "val_loss": chosen["best"]["val_loss"]}, f, indent=2)

    # ---- tables and raw histories
    write_table2(chosen, init_loss, n_params, device, "results/table2_network_training.md")
    write_table3(runs, "results/table3_lr_choice.md")
    with open("results/history.json", "w") as f:
        json.dump({str(r["alpha"]): {**r["history"], "best_epoch": r["best"]["epoch"],
                                     "seconds": r["seconds"], "diverged": r["diverged"]} for r in runs},
                  f, indent=2)

    print("\nsaved: results/lr_val_loss.png, loss_curves.png, accuracy_curves.png, ecgnet_best.pt,")
    print("       table2_network_training.md, table3_lr_choice.md, history.json, best_model_info.json")


if __name__ == "__main__":
    main()

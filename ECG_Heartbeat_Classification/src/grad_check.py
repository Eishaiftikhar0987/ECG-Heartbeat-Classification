"""Task 3.2 - Gradient check of the output layer.

Run from the repository root:   python src/grad_check.py

dJ/dW[3] = (1/B) (Yhat - Y)^T A[2]        dJ/db[3] = (1/B) sum_i (yhat_i - y_i)
is computed by hand and compared with what loss.backward() stored in .grad.
"""
import json
import os

import torch

from model import cross_entropy, make_model, softmax_probs
from prepare import prepare

B = 64
TOLERANCE = 1e-5


def run_grad_check(data):
    model = make_model(0)                       # fresh network, seed 0
    x, y = data["X_train"][:B], data["y_train"][:B]   # first 64 training rows

    # --- PyTorch (autograd) gradient
    logits, acts = model(x, return_activations=True)
    loss = cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()
    auto_dW3 = model.W[2].grad.clone()
    auto_db3 = model.b[2].grad.clone()

    # --- Gradient computed by hand
    with torch.no_grad():
        Y_hat = softmax_probs(logits)                                    # (B, 5)
        Y = torch.zeros(B, 5).scatter_(1, y.unsqueeze(1), 1.0)           # one-hot (B, 5)
        A2 = acts[2]                                                     # (B, 64)
        hand_dW3 = (Y_hat - Y).T @ A2 / B                                # (5, 64)
        hand_db3 = (Y_hat - Y).sum(dim=0) / B                            # (5,)

    diff_W = (hand_dW3 - auto_dW3).abs().max().item()
    diff_b = (hand_db3 - auto_db3).abs().max().item()
    return {
        "batch_size": B,
        "loss_on_batch": loss.item(),
        "max_abs_diff_W3": diff_W,
        "max_abs_diff_b3": diff_b,
        "tolerance": TOLERANCE,
        "passed": bool(diff_W < TOLERANCE and diff_b < TOLERANCE),
    }


if __name__ == "__main__":
    data, _ = prepare(verbose=False)
    res = run_grad_check(data)
    print(f"loss on the first {B} training rows (fresh network): {res['loss_on_batch']:.6f}")
    print(f"max |hand - autograd| for W[3]: {res['max_abs_diff_W3']:.3e}")
    print(f"max |hand - autograd| for b[3]: {res['max_abs_diff_b3']:.3e}")
    print("PASS (difference < 1e-5)" if res["passed"] else "FAIL - check your gradient formula")
    os.makedirs("results", exist_ok=True)
    with open("results/grad_check.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved results/grad_check.json")

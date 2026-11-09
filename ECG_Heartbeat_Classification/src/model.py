"""Task 2 (network) and Task 3.1 (loss).

Notation follows the lecture notes: layer l has weight matrix W[l] of shape
(n[l], n[l-1]) and bias b[l].  In the code the lists are 0-based, so
    model.W[0] = W[1],  model.W[1] = W[2],  model.W[2] = W[3]  (same for b).

Only torch tensors, nn.Module, nn.Parameter and nn.ParameterList are used - no
nn.Linear / nn.Sequential / nn.ReLU / nn.init / CrossEntropyLoss anywhere.
"""
import math

import torch
from torch import nn

LAYER_SIZES = (187, 128, 64, 5)   # input, hidden 1, hidden 2, output


def relu(z):
    """g(z) = max(z, 0), written by hand."""
    return torch.clamp(z, min=0)


class ECGNet(nn.Module):
    def __init__(self, sizes=LAYER_SIZES):
        super().__init__()
        self.sizes = tuple(sizes)
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for l in range(1, len(sizes)):
            n_in, n_out = sizes[l - 1], sizes[l]
            # He initialisation: W_ij ~ N(0, sqrt(2 / n[l-1])), b = 0
            std = math.sqrt(2.0 / n_in)
            self.W.append(nn.Parameter(torch.randn(n_out, n_in) * std))
            self.b.append(nn.Parameter(torch.zeros(n_out)))

    def forward(self, x, return_activations=False):
        """x has shape (B, 187). Returns the logits z[3] of shape (B, 5).

        With one example per row a layer is  Z[l] = A[l-1] @ W[l].T + b[l].
        If return_activations is True, also returns [A0, A1, A2] (needed by the
        gradient check in Task 3.2).
        """
        a = x
        activations = [a]
        last = len(self.W) - 1
        for l in range(len(self.W)):
            z = a @ self.W[l].T + self.b[l]
            if l < last:
                a = relu(z)
                activations.append(a)
            else:
                logits = z          # output layer: no activation (softmax is in the loss)
        if return_activations:
            return logits, activations
        return logits


def make_model(seed=0):
    """Fresh network with the assignment's seed (torch.manual_seed(0) before creating)."""
    torch.manual_seed(seed)
    return ECGNet()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_count_by_hand(sizes=LAYER_SIZES):
    """Return (text, total) showing the hand calculation  n[l] * n[l-1] + n[l]  per layer."""
    lines, total = [], 0
    for l in range(1, len(sizes)):
        n_in, n_out = sizes[l - 1], sizes[l]
        w, b = n_out * n_in, n_out
        lines.append(f"Layer {l}: {n_out} x {n_in} weights + {n_out} biases = {w:,} + {b:,} = {w + b:,}")
        total += w + b
    lines.append("Total = " + " + ".join(f"{(sizes[l] * sizes[l - 1] + sizes[l]):,}" for l in range(1, len(sizes)))
                 + f" = {total:,}")
    return "\n".join(lines), total


def cross_entropy(logits, y):
    """Mean softmax cross-entropy over the batch.

    L = -z_y + log(sum_j exp(z_j)), with torch.logsumexp for numerical stability.
    logits: (B, 5) float tensor, y: (B,) long tensor with values 0..4.
    """
    lse = torch.logsumexp(logits, dim=1)                       # (B,)
    true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)   # z_y, shape (B,)
    return (lse - true_logit).mean()


def softmax_probs(logits):
    """Class probabilities exp(z_k) / sum_j exp(z_j), computed stably (no torch.softmax)."""
    return torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))


if __name__ == "__main__":
    model = make_model(0)
    print(model)
    text, hand_total = parameter_count_by_hand()
    print(text)
    n = count_parameters(model)
    print(f"count_parameters(model) = {n:,}")
    assert n == hand_total == 32645, "parameter count should be 32,645"
    print("parameter count check: OK (32,645)")

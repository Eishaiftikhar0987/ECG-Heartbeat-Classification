"""Task 5.2 - the metric functions, written by hand (numpy only, no scikit-learn)."""
import numpy as np


def accuracy(y_true, y_pred):
    """Share of rows whose predicted class equals the true class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sum(y_true == y_pred)) / len(y_true)


def confusion_matrix(y_true, y_pred, n_classes=5):
    """C[j, k] = number of rows whose true class is j and predicted class is k."""
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        C[int(t), int(p)] += 1
    return C


def per_class_accuracy(C):
    """For class j: C[j, j] / sum_k C[j, k] (share of class-j beats recognised as class j)."""
    return np.array([C[j, j] / C[j].sum() for j in range(C.shape[0])])


def balanced_accuracy(C):
    """Average of the per-class accuracies - every class counts equally."""
    return float(per_class_accuracy(C).mean())

"""
Loss / Objective Functions and Their Gradients
Supports: cross_entropy, mse
"""
import numpy as np
from ann.activations import softmax


def cross_entropy_loss(logits: np.ndarray, y_onehot: np.ndarray) -> float:
    """
    Cross-entropy loss computed from raw logits (softmax applied internally).

    Args:
        logits:   (N, C) unnormalised scores
        y_onehot: (N, C) one-hot targets
    Returns:
        Scalar mean loss.
    """
    probs = np.clip(softmax(logits), 1e-12, 1.0)
    return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))


def cross_entropy_grad(logits: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """
    dL/d(logits) = (softmax(logits) - y) / N
    Returns:
        (N, C) gradient array.
    """
    N = logits.shape[0]
    return (softmax(logits) - y_onehot) / N


def mse_loss(logits: np.ndarray, y_onehot: np.ndarray) -> float:
    """MSE loss with softmax applied to logits."""
    probs = softmax(logits)
    return np.mean(np.sum((probs - y_onehot) ** 2, axis=1))


def mse_grad(logits: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE w.r.t. logits via chain rule through softmax.
    Returns:
        (N, C) gradient array.
    """
    N = logits.shape[0]
    probs = softmax(logits)
    dL_dp = 2.0 * (probs - y_onehot) / N
    dot = np.sum(dL_dp * probs, axis=1, keepdims=True)
    return probs * (dL_dp - dot)


def get_loss(name: str):
    """Return (loss_fn, grad_fn) pair by name."""
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("cross_entropy",):
        return cross_entropy_loss, cross_entropy_grad
    elif name in ("mse", "mean_squared_error"):
        return mse_loss, mse_grad
    else:
        raise ValueError(f"Unknown loss '{name}'. Choose from: cross_entropy, mse.")

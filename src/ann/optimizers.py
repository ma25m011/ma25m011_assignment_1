"""
Optimisation Algorithms
Implements: SGD, Momentum, NAG, RMSProp
"""
import numpy as np
from typing import List, Dict


class SGD:
    """Vanilla SGD with optional L2 weight decay."""

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, params: List[Dict], grads: List[Dict]):
        for p, g in zip(params, grads):
            p["W"] -= self.lr * (g["grad_W"] + self.weight_decay * p["W"])
            p["b"] -= self.lr * g["grad_b"]


def _shapes_match(state: List[Dict], params: List[Dict]) -> bool:
    if len(state) != len(params):
        return False
    return all(
        s["W"].shape == p["W"].shape and s["b"].shape == p["b"].shape
        for s, p in zip(state, params)
    )


class Momentum:
    """SGD with classical momentum."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v: List[Dict] = []

    def update(self, params: List[Dict], grads: List[Dict]):
        if not self.v or not _shapes_match(self.v, params):
            self.v = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])} for p in params]
        for p, g, v in zip(params, grads, self.v):
            v["W"] = self.beta * v["W"] - self.lr * (g["grad_W"] + self.weight_decay * p["W"])
            v["b"] = self.beta * v["b"] - self.lr * g["grad_b"]
            p["W"] += v["W"]
            p["b"] += v["b"]


class NAG:
    """Nesterov Accelerated Gradient."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v: List[Dict] = []

    def _init_state(self, params):
        self.v = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])} for p in params]

    def update(self, params: List[Dict], grads: List[Dict]):
        if not self.v or not _shapes_match(self.v, params):
            self._init_state(params)
        for p, g, v in zip(params, grads, self.v):
            v["W"] = self.beta * v["W"] - self.lr * (g["grad_W"] + self.weight_decay * p["W"])
            v["b"] = self.beta * v["b"] - self.lr * g["grad_b"]
            p["W"] += v["W"]
            p["b"] += v["b"]

    def apply_lookahead(self, params: List[Dict]):
        """Shift weights to the lookahead position before computing gradients."""
        if not self.v or not _shapes_match(self.v, params):
            self._init_state(params)
        for p, v in zip(params, self.v):
            p["W"] += self.beta * v["W"]
            p["b"] += self.beta * v["b"]

    def restore_weights(self, params: List[Dict]):
        """Undo the lookahead shift after gradient computation."""
        for p, v in zip(params, self.v):
            p["W"] -= self.beta * v["W"]
            p["b"] -= self.beta * v["b"]


class RMSProp:
    """RMSProp optimiser."""

    def __init__(self, lr: float = 0.001, beta: float = 0.9,
                 epsilon: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.sq: List[Dict] = []

    def update(self, params: List[Dict], grads: List[Dict]):
        if not self.sq or not _shapes_match(self.sq, params):
            self.sq = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])} for p in params]
        for p, g, sq in zip(params, grads, self.sq):
            gW = g["grad_W"] + self.weight_decay * p["W"]
            gb = g["grad_b"]
            sq["W"] = self.beta * sq["W"] + (1.0 - self.beta) * gW ** 2
            sq["b"] = self.beta * sq["b"] + (1.0 - self.beta) * gb ** 2
            p["W"] -= self.lr * gW / (np.sqrt(sq["W"]) + self.epsilon)
            p["b"] -= self.lr * gb / (np.sqrt(sq["b"]) + self.epsilon)


def get_optimizer(name: str, lr: float, weight_decay: float = 0.0, **kwargs):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=lr, weight_decay=weight_decay)
    elif name == "momentum":
        return Momentum(lr=lr, weight_decay=weight_decay, beta=kwargs.get("beta", 0.9))
    elif name == "nag":
        return NAG(lr=lr, weight_decay=weight_decay, beta=kwargs.get("beta", 0.9))
    elif name == "rmsprop":
        return RMSProp(lr=lr, weight_decay=weight_decay,
                       beta=kwargs.get("beta", 0.9), epsilon=kwargs.get("epsilon", 1e-8))
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: sgd, momentum, nag, rmsprop.")

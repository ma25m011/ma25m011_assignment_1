"""
Neural Layer Implementation
Handles weight initialisation, forward pass, and gradient computation.
"""
import numpy as np
from typing import Optional
from ann.activations import get_activation


class Layer:
    """
    Single fully-connected hidden layer.

    After backward() is called the following are available:
        self.grad_W  –  gradient w.r.t. W  (same shape as W)
        self.grad_b  –  gradient w.r.t. b  (same shape as b)
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu", weight_init: str = "xavier"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation

        self.act_fn, self.act_grad = get_activation(activation)

        self.W = self._init_weights(in_features, out_features, weight_init)
        self.b = np.zeros((1, out_features))

        self.grad_W: Optional[np.ndarray] = None
        self.grad_b: Optional[np.ndarray] = None
        self._input: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None

    @staticmethod
    def _init_weights(fan_in, fan_out, scheme):
        scheme = scheme.lower()
        if scheme == "xavier":
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            return np.random.randn(fan_in, fan_out) * scale
        elif scheme == "random":
            return np.random.randn(fan_in, fan_out) * 0.01
        elif scheme == "zeros":
            return np.zeros((fan_in, fan_out))
        else:
            raise ValueError(f"Unknown weight_init: {scheme}")

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        self._input = a_prev
        self._z = a_prev @ self.W + self.b
        return self.act_fn(self._z)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        assert self._input is not None, "forward() must be called before backward()"
        dz = delta * self.act_grad(self._z)
        self.grad_W = self._input.T @ dz
        self.grad_b = dz.sum(axis=0, keepdims=True)
        return dz @ self.W.T

    def get_params(self):
        return {"W": self.W, "b": self.b}

    def get_grads(self):
        return {"grad_W": self.grad_W, "grad_b": self.grad_b}


class OutputLayer:
    """
    Linear output layer — returns raw logits.
    Softmax is applied inside the loss functions, not here.
    """

    def __init__(self, in_features: int, out_features: int,
                 weight_init: str = "xavier"):
        self.in_features = in_features
        self.out_features = out_features

        self.W = Layer._init_weights(in_features, out_features, weight_init)
        self.b = np.zeros((1, out_features))

        self.grad_W: Optional[np.ndarray] = None
        self.grad_b: Optional[np.ndarray] = None
        self._input: Optional[np.ndarray] = None

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        self._input = a_prev
        return a_prev @ self.W + self.b

    def backward(self, delta: np.ndarray) -> np.ndarray:
        assert self._input is not None, "forward() must be called before backward()"
        self.grad_W = self._input.T @ delta
        self.grad_b = delta.sum(axis=0, keepdims=True)
        return delta @ self.W.T

    def get_params(self):
        return {"W": self.W, "b": self.b}

    def get_grads(self):
        return {"grad_W": self.grad_W, "grad_b": self.grad_b}

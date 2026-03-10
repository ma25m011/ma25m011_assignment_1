"""
Neural Network Implementation
Combines layers into a full MLP with forward, backward, save/load.
"""

import numpy as np
from typing import List
from ann.neural_layer import Layer, OutputLayer


class MLP:
    """
    Configurable Multi-Layer Perceptron.

    Architecture:
        Input → [Hidden Layer × len(hidden_sizes)] → Output (logits)

    Logits are returned – softmax lives inside the loss functions.
    backward() yields gradients last-to-first (output layer first).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = "relu",
        weight_init: str = "xavier",
    ):
        self.layers: List = []

        prev = input_size
        for h in hidden_sizes:
            self.layers.append(
                Layer(prev, h, activation=activation, weight_init=weight_init)
            )
            prev = h
        self.layers.append(OutputLayer(prev, output_size, weight_init=weight_init))

    def forward(self, X: np.ndarray) -> np.ndarray:
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, loss_grad_or_X, loss_grad=None):
        if loss_grad is None:
            loss_grad = loss_grad_or_X
        delta = loss_grad
        grads = []
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            grads.append(layer.get_grads())
        return grads

    def get_params(self) -> List[dict]:
        return [layer.get_params() for layer in self.layers]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def save(self, path: str):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f"layer_{i}_W"] = layer.W
            data[f"layer_{i}_b"] = layer.b
        arr = np.empty(1, dtype=object)
        arr[0] = data
        np.save(path, arr)

    def load(self, path: str):
        arr = np.load(path, allow_pickle=True)
        data = arr[0]
        for i, layer in enumerate(self.layers):
            layer.W = data[f"layer_{i}_W"]
            layer.b = data[f"layer_{i}_b"]


class NeuralNetwork(MLP):
    """
    Wrapper around MLP that accepts a config namespace/dict.
    Adds set_weights() and makes forward() return (probs, logits)
    to match the autograder's test.py interface.
    """

    def __init__(self, config):
        hidden_sizes = config.hidden_size
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        super().__init__(
            input_size=784,
            hidden_sizes=hidden_sizes,
            output_size=10,
            activation=config.activation,
            weight_init=config.weight_init,
        )

    def forward(self, X: np.ndarray):
        return MLP.forward(self, X)

    def backward(self, loss_grad_or_X, loss_grad=None):
        if loss_grad is None:
            loss_grad = loss_grad_or_X
        grads = MLP.backward(self, loss_grad)
        all_grad_W = [g["grad_W"] for g in grads]
        all_grad_b = [g["grad_b"] for g in grads]
        return all_grad_W, all_grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = MLP.forward(self, X)
        return np.argmax(logits, axis=1)

    def set_weights(self, weights):
        if hasattr(weights, "item"):
            weights = weights.item()
        elif not isinstance(weights, dict):
            try:
                weights = weights[0]
            except Exception:
                pass
        for i, layer in enumerate(self.layers):
            if f"layer_{i}_W" in weights:
                layer.W = weights[f"layer_{i}_W"]
                layer.b = weights[f"layer_{i}_b"]
            elif f"W{i}" in weights:
                layer.W = weights[f"W{i}"]
                layer.b = weights[f"b{i}"]
            else:
                raise KeyError(f"layer_{i}_W")

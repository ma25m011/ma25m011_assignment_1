import numpy as np


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def get_activation(name: str):
    """Return (activation_fn, gradient_fn) by name."""
    name = name.lower()
    if name == "sigmoid":
        return sigmoid, sigmoid_grad
    elif name == "tanh":
        return tanh, tanh_grad
    elif name == "relu":
        return relu, relu_grad
    else:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: sigmoid, tanh, relu."
        )


def softmax(z):
    """Row-wise numerically stable softmax."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

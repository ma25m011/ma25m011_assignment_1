import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset: str = "fashion_mnist", val_split: float = 0.1):
    """
    Load MNIST or Fashion-MNIST via keras.datasets.

    Returns:
        X_train, X_val, X_test : (N, 784) float32, values in [0, 1]
        y_train, y_val, y_test  : (N,) int labels
    """
    dataset = dataset.lower().replace("-", "_").replace(" ", "_")

    if dataset == "mnist":
        from tensorflow.keras.datasets import mnist
        (X_raw, y_raw), (X_test, y_test) = mnist.load_data()
    elif dataset in ("fashion_mnist", "fashionmnist"):
        from tensorflow.keras.datasets import fashion_mnist
        (X_raw, y_raw), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: mnist, fashion_mnist.")

    X_raw  = X_raw.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=val_split, random_state=42, stratify=y_raw
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_onehot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    N = y.shape[0]
    onehot = np.zeros((N, num_classes), dtype=np.float32)
    onehot[np.arange(N), y] = 1.0
    return onehot


def get_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """Yields (X_batch, y_batch) mini-batches."""
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], y[idx]

import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ann.neural_network import MLP
from utils.data_loader import load_data
from utils.metrics import compute_metrics, get_confusion_matrix


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference with a saved MLP checkpoint"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="fashion_mnist",
        choices=["mnist", "fashion_mnist"],
    )
    parser.add_argument("-e", "--epochs", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "mse"],
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="rmsprop",
        choices=["sgd", "momentum", "nag", "rmsprop"],
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument(
        "-sz", "--hidden_size", type=int, nargs="+", default=[128, 128, 64]
    )
    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        default="relu",
        choices=["sigmoid", "tanh", "relu"],
    )
    parser.add_argument(
        "-w_i",
        "--weight_init",
        type=str,
        default="xavier",
        choices=["random", "xavier"],
    )
    parser.add_argument("-w_p", "--wandb_project", type=str, default=None)
    parser.add_argument(
        "--model_path", type=str, default=os.path.join(SRC, "best_model.npy")
    )
    parser.add_argument(
        "--config_path", type=str, default=os.path.join(SRC, "best_config.json")
    )
    return parser.parse_args()


def run_inference(args):
    # load saved config if present, fall back to CLI args
    cfg = {}
    if os.path.exists(args.config_path):
        with open(args.config_path) as f:
            cfg = json.load(f)
        print(f"loaded config from {args.config_path}")

    hidden_sizes = cfg.get("hidden_sizes", args.hidden_size)
    activation = cfg.get("activation", args.activation)
    weight_init = cfg.get("weight_init", args.weight_init)
    dataset = cfg.get("dataset", args.dataset)
    n_classes = cfg.get("output_size", 10)

    model = MLP(784, hidden_sizes, n_classes, activation, weight_init)
    model.load(args.model_path)
    print(f"model loaded from {args.model_path}")

    _, _, X_test, _, _, y_test = load_data(dataset)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    cm = get_confusion_matrix(y_test, y_pred)

    print("\n── test results ──────────────────────")
    print(f"  accuracy  : {metrics['accuracy']:.4f}")
    print(f"  precision : {metrics['precision']:.4f}")
    print(f"  recall    : {metrics['recall']:.4f}")
    print(f"  f1-score  : {metrics['f1']:.4f}")
    print("\nconfusion matrix:")
    print(cm)

    return metrics


def get_args():
    return parse_arguments()


if __name__ == "__main__":
    run_inference(parse_arguments())

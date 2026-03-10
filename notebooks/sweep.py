import argparse
import os
import sys
import numpy as np

# allow running as python wandb/sweep.py from repo root
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import wandb
from ann.neural_network import MLP
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer, NAG
from utils.data_loader import load_data, to_onehot, get_batches
from utils.metrics import compute_metrics


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "batch_size": {"values": [32, 64, 128]},
        "num_layers": {"values": [1, 2, 3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "weight_decay": {"values": [0.0, 1e-4, 1e-3]},
        "weight_init": {"values": ["xavier", "random"]},
        "loss": {"values": ["cross_entropy", "mse"]},
        "epochs": {"value": 10},
    },
}

NUM_CLASSES = 10


def sweep_train():
    run = wandb.init()
    cfg = wandb.config

    X_train, X_val, X_test, y_train, y_val, y_test = load_data("fashion_mnist")
    hidden_sizes = [cfg.hidden_size] * cfg.num_layers

    model = MLP(
        input_size=784,
        hidden_sizes=hidden_sizes,
        output_size=NUM_CLASSES,
        activation=cfg.activation,
        weight_init=cfg.weight_init,
    )

    loss_fn, loss_grad_fn = get_loss(cfg.loss)
    optimizer = get_optimizer(
        cfg.optimizer, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    best_val_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        losses = []
        for X_batch, y_batch in get_batches(X_train, y_train, cfg.batch_size):
            y_oh = to_onehot(y_batch, NUM_CLASSES)

            if isinstance(optimizer, NAG):
                optimizer.apply_lookahead(model.get_params())

            logits = model.forward(X_batch)
            loss = loss_fn(logits, y_oh)
            grad = loss_grad_fn(logits, y_oh)

            if isinstance(optimizer, NAG):
                optimizer.restore_weights(model.get_params())

            grads = model.backward(grad)
            optimizer.update(
                [layer.get_params() for layer in reversed(model.layers)], grads
            )
            losses.append(loss)

        val_preds = model.predict(X_val)
        val_metrics = compute_metrics(y_val, val_preds)
        train_acc = compute_metrics(y_train, model.predict(X_train))["accuracy"]

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_acc": train_acc,
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]

    wandb.log({"best_val_f1": best_val_f1})
    run.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wandb_project", type=str, required=True)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--count", type=int, default=100)
    args = p.parse_args()

    sweep_id = wandb.sweep(
        SWEEP_CONFIG, project=args.wandb_project, entity=args.wandb_entity
    )
    wandb.agent(sweep_id, function=sweep_train, count=args.count)


if __name__ == "__main__":
    main()

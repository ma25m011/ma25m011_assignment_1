import argparse
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, ".."))
_src = os.path.abspath(os.path.join(_root, "src"))
for _p in (_root, _src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wandb
from ann.neural_network import MLP
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer, NAG
from utils.data_loader import load_data, to_onehot, get_batches
from utils.metrics import compute_metrics, get_confusion_matrix


NUM_CLASSES = 10
FASHION_LABELS = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
MNIST_LABELS = [str(i) for i in range(10)]


def quick_train(
    hidden_sizes,
    activation,
    optimizer_name,
    lr,
    epochs,
    batch_size,
    loss_name,
    weight_init,
    X_train,
    y_train,
    X_val,
    y_val,
    weight_decay=0.0,
    log_grad_norms=False,
):
    """Train a fresh MLP and return (model, history_dict)."""
    model = MLP(784, hidden_sizes, NUM_CLASSES, activation, weight_init)
    loss_fn, loss_grad_fn = get_loss(loss_name)
    optimizer = get_optimizer(optimizer_name, lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "grad_norms": []}

    for epoch in range(1, epochs + 1):
        losses, grad_norms_epoch = [], []

        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            y_oh = to_onehot(y_batch, NUM_CLASSES)

            if isinstance(optimizer, NAG):
                optimizer.apply_lookahead(model.get_params())

            logits = model.forward(X_batch)
            loss = loss_fn(logits, y_oh)
            grad = loss_grad_fn(logits, y_oh)

            if isinstance(optimizer, NAG):
                optimizer.restore_weights(model.get_params())

            grads = model.backward(grad)

            if log_grad_norms:
                first_grad = grads[-1]["grad_W"]
                grad_norms_epoch.append(float(np.linalg.norm(first_grad)))

            optimizer.update(
                [layer.get_params() for layer in reversed(model.layers)], grads
            )
            losses.append(loss)

        val_preds = model.predict(X_val)
        val_acc = compute_metrics(y_val, val_preds)["accuracy"]
        train_preds = model.predict(X_train)
        train_acc = compute_metrics(y_train, train_preds)["accuracy"]

        history["train_loss"].append(float(np.mean(losses)))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if log_grad_norms:
            history["grad_norms"].append(float(np.mean(grad_norms_epoch)))

    return model, history


# ─── Section 2.1  Data Exploration ────────────────────────────────────


def section_2_1(X_train, y_train, dataset):
    labels = FASHION_LABELS if "fashion" in dataset else MNIST_LABELS
    print("2.1 – Logging sample images table …")

    columns = ["label"] + [f"image_{i}" for i in range(5)]
    table = wandb.Table(columns=columns)

    for cls in range(NUM_CLASSES):
        idx = np.where(y_train == cls)[0][:5]
        images = [wandb.Image(X_train[i].reshape(28, 28)) for i in idx]
        table.add_data(labels[cls], *images)

    wandb.log({"2.1_sample_images": table})
    print("  ✓ done")


# ─── Section 2.2  Hyperparameter Sweep ────────────────────────────────


def section_2_2(project, entity, X_train, y_train, X_val, y_val):
    """
    Run a Bayesian sweep (≥100 runs) optimising val_f1.
    This section runs its own sweep agent and logs results back to W&B.
    """
    print("2.2 – Hyperparameter sweep (100 runs) …")

    sweep_config = {
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

    def sweep_run():
        with wandb.init() as run:
            cfg = wandb.config
            hidden_sizes = [cfg.hidden_size] * cfg.num_layers

            model = MLP(784, hidden_sizes, NUM_CLASSES, cfg.activation, cfg.weight_init)
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

                val_m = compute_metrics(y_val, model.predict(X_val))
                train_acc = compute_metrics(y_train, model.predict(X_train))["accuracy"]

                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "train_acc": train_acc,
                        "val_acc": val_m["accuracy"],
                        "val_f1": val_m["f1"],
                    }
                )
                if val_m["f1"] > best_val_f1:
                    best_val_f1 = val_m["f1"]

            wandb.log({"best_val_f1": best_val_f1})

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    wandb.agent(sweep_id, function=sweep_run, count=100)
    print("  ✓ done")


# ─── Section 2.3  Optimizer Showdown ──────────────────────────────────


def section_2_3(X_train, y_train, X_val, y_val):
    print("2.3 – Optimizer showdown …")
    optimizers = ["sgd", "momentum", "nag", "rmsprop"]
    lrs = {"sgd": 1e-2, "momentum": 1e-2, "nag": 1e-2, "rmsprop": 1e-3}
    epochs = 15

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for opt in optimizers:
        _, hist = quick_train(
            [128, 64],
            "relu",
            opt,
            lrs[opt],
            epochs,
            64,
            "cross_entropy",
            "xavier",
            X_train,
            y_train,
            X_val,
            y_val,
        )
        axes[0].plot(hist["train_loss"], label=opt)
        axes[1].plot(hist["val_acc"], label=opt)

    for ax, title, ylabel in zip(
        axes, ["Train Loss", "Val Accuracy"], ["Loss", "Accuracy"]
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({"2.3_optimizer_showdown": wandb.Image(fig)})
    plt.close(fig)
    print("  ✓ done")


# ─── Section 2.4  Vanishing Gradient Analysis ─────────────────────────


def section_2_4(X_train, y_train, X_val, y_val):
    print("2.4 – Vanishing gradient analysis …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for act, ax in zip(["sigmoid", "relu"], axes):
        _, hist = quick_train(
            [128, 128, 128, 64],
            act,
            "rmsprop",
            1e-3,
            15,
            64,
            "cross_entropy",
            "xavier",
            X_train,
            y_train,
            X_val,
            y_val,
            log_grad_norms=True,
        )
        ax.plot(hist["grad_norms"])
        ax.set_title(f"Gradient Norms – {act.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L2 norm (first hidden layer)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({"2.4_vanishing_gradient": wandb.Image(fig)})
    plt.close(fig)
    print("  ✓ done")


# ─── Section 2.5  Dead Neuron Investigation ───────────────────────────


def section_2_5(X_train, y_train, X_val, y_val):
    print("2.5 – Dead neuron investigation …")
    _, hist_high = quick_train(
        [128, 128],
        "relu",
        "sgd",
        5e-1,
        15,
        64,
        "cross_entropy",
        "random",
        X_train,
        y_train,
        X_val,
        y_val,
        log_grad_norms=True,
    )
    _, hist_low = quick_train(
        [128, 128],
        "relu",
        "sgd",
        1e-3,
        15,
        64,
        "cross_entropy",
        "xavier",
        X_train,
        y_train,
        X_val,
        y_val,
        log_grad_norms=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for hist, label, ax in zip(
        [hist_high, hist_low],
        ["High LR (SGD, lr=0.5)", "Low LR (SGD, lr=0.001)"],
        axes,
    ):
        ax.plot(hist["train_acc"], label="Train Acc")
        ax.plot(hist["val_acc"], label="Val Acc")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({"2.5_dead_neuron": wandb.Image(fig)})
    plt.close(fig)
    print("  ✓ done")


# ─── Section 2.6  Loss Function Comparison ────────────────────────────


def section_2_6(X_train, y_train, X_val, y_val):
    print("2.6 – MSE vs Cross-Entropy …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for loss_name, ax in zip(["cross_entropy", "mse"], axes):
        _, hist = quick_train(
            [128, 64],
            "relu",
            "rmsprop",
            1e-3,
            15,
            64,
            loss_name,
            "xavier",
            X_train,
            y_train,
            X_val,
            y_val,
        )
        ax.plot(hist["train_loss"], label="Train Loss")
        ax.plot(hist["val_acc"], label="Val Acc", linestyle="--")
        ax.set_title(f"Loss: {loss_name}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({"2.6_loss_comparison": wandb.Image(fig)})
    plt.close(fig)
    print("  ✓ done")


# ─── Section 2.7  Global Performance Analysis ─────────────────────────


def section_2_7(X_train, y_train, X_val, y_val):
    print("2.7 – Global performance (train vs val) …")
    _, hist = quick_train(
        [128, 128, 64],
        "relu",
        "rmsprop",
        1e-3,
        20,
        64,
        "cross_entropy",
        "xavier",
        X_train,
        y_train,
        X_val,
        y_val,
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["train_acc"], label="Train Accuracy")
    ax.plot(hist["val_acc"], label="Val Accuracy", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Val Accuracy over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({"2.7_train_val_accuracy": wandb.Image(fig)})
    plt.close(fig)
    print("  ✓ done")


# ─── Section 2.8  Error Analysis ──────────────────────────────────────


def section_2_8(X_train, y_train, X_val, y_val, X_test, y_test, dataset):
    print("2.8 – Confusion matrix …")
    labels = FASHION_LABELS if "fashion" in dataset else MNIST_LABELS

    model, _ = quick_train(
        [128, 128, 64],
        "relu",
        "rmsprop",
        1e-3,
        20,
        64,
        "cross_entropy",
        "xavier",
        X_train,
        y_train,
        X_val,
        y_val,
    )
    y_pred = model.predict(X_test)
    cm = get_confusion_matrix(y_test, y_pred)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    wandb.log({"2.8_confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    # Creative: per-class accuracy bar chart coloured by performance
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    cmap = plt.colormaps.get_cmap("RdYlGn")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bars = ax2.bar(labels, per_class_acc, color=[cmap(v) for v in per_class_acc])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Per-class Accuracy")
    ax2.set_title("Per-class Accuracy (green = good, red = bad)")
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    for bar, val in zip(bars, per_class_acc):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.2f}",
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    wandb.log({"2.8_per_class_accuracy": wandb.Image(fig2)})
    plt.close(fig2)
    print("  ✓ done")


# ─── Section 2.9  Weight Initialisation ───────────────────────────────


def section_2_9(X_train, y_train, X_val, y_val):
    print("2.9 – Weight initialisation comparison …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for init, ax in zip(["xavier", "zeros"], axes):
        _, hist = quick_train(
            [128, 128, 64],
            "relu",
            "rmsprop",
            1e-3,
            20,
            64,
            "cross_entropy",
            init,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        ax.plot(hist["train_loss"], label="Train Loss")
        ax.plot(hist["val_acc"], label="Val Acc", linestyle="--")
        ax.set_title(f"Init: {init}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({"2.9_weight_init": wandb.Image(fig)})
    plt.close(fig)
    print("  ✓ done")


# ─── Section 2.10  Fashion-MNIST Transfer ─────────────────────────────


def section_2_10(X_train, y_train, X_val, y_val, X_test, y_test):
    print("2.10 – Fashion-MNIST transfer challenge …")
    configs = [
        {
            "name": "Deep+ReLU+RMSProp",
            "hidden": [128, 128, 128, 64],
            "act": "relu",
            "opt": "rmsprop",
            "lr": 1e-3,
        },
        {
            "name": "Shallow+Tanh+Momentum",
            "hidden": [256, 128],
            "act": "tanh",
            "opt": "momentum",
            "lr": 5e-3,
        },
        {
            "name": "Wide+Sigmoid+NAG",
            "hidden": [512],
            "act": "sigmoid",
            "opt": "nag",
            "lr": 5e-3,
        },
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for cfg in configs:
        model, hist = quick_train(
            cfg["hidden"],
            cfg["act"],
            cfg["opt"],
            cfg["lr"],
            20,
            64,
            "cross_entropy",
            "xavier",
            X_train,
            y_train,
            X_val,
            y_val,
        )
        test_m = compute_metrics(y_test, model.predict(X_test))
        label = f"{cfg['name']} | test_f1={test_m['f1']:.3f}"
        ax.plot(hist["val_acc"], label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Fashion-MNIST – 3 Configurations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({"2.10_fashion_configs": wandb.Image(fig)})
    plt.close(fig)
    print("  ✓ done")


# ─── Main ──────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wandb_project", type=str, required=True)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--dataset", type=str, default="fashion_mnist")
    p.add_argument(
        "--skip_sweep",
        action="store_true",
        help="Skip section 2.2 sweep (useful for quick testing)",
    )
    args = p.parse_args()

    print(f"Loading {args.dataset} …")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.dataset)

    # Sections 2.3–2.10 all log into a single analysis run
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="analysis_all_sections",
        job_type="analysis",
    )

    section_2_1(X_train, y_train, args.dataset)
    section_2_3(X_train, y_train, X_val, y_val)
    section_2_4(X_train, y_train, X_val, y_val)
    section_2_5(X_train, y_train, X_val, y_val)
    section_2_6(X_train, y_train, X_val, y_val)
    section_2_7(X_train, y_train, X_val, y_val)
    section_2_8(X_train, y_train, X_val, y_val, X_test, y_test, args.dataset)
    section_2_9(X_train, y_train, X_val, y_val)
    section_2_10(X_train, y_train, X_val, y_val, X_test, y_test)

    run.finish()

    # Section 2.2 runs its own sweep agent (separate runs), do it last
    if not args.skip_sweep:
        section_2_2(
            args.wandb_project, args.wandb_entity, X_train, y_train, X_val, y_val
        )

    print("\nAll W&B sections logged successfully!")


if __name__ == "__main__":
    main()

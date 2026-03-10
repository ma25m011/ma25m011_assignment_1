import argparse
import json
import os
import sys

import numpy as np

from ann.neural_network import MLP
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer, NAG
from utils.data_loader import load_data, to_onehot, get_batches
from utils.metrics import compute_metrics

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a NumPy MLP on MNIST / Fashion-MNIST"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="fashion_mnist",
        choices=["mnist", "fashion_mnist"],
    )
    parser.add_argument("-e", "--epochs", type=int, default=10)
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
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--model_save_path", type=str, default=os.path.join(SRC, "best_model.npy")
    )
    parser.add_argument(
        "--config_save_path", type=str, default=os.path.join(SRC, "best_config.json")
    )
    return parser.parse_args()


def train(args):
    # resolve hidden layer sizes
    sizes = list(args.hidden_size)
    if len(sizes) == 1:
        sizes = sizes * args.num_layers
    elif len(sizes) < args.num_layers:
        sizes = sizes + [sizes[-1]] * (args.num_layers - len(sizes))
    else:
        sizes = sizes[: args.num_layers]

    use_wandb = (args.wandb_project is not None) and (not args.no_wandb)
    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, config=vars(args)
        )

    print(f"loading {args.dataset} ...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.dataset)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    n_classes = 10
    model = MLP(784, sizes, n_classes, args.activation, args.weight_init)
    loss_fn, grad_fn = get_loss(args.loss)
    opt = get_optimizer(
        args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_f1 = -1.0
    os.makedirs(os.path.dirname(os.path.abspath(args.model_save_path)), exist_ok=True)

    for ep in range(1, args.epochs + 1):
        batch_losses = []

        for xb, yb in get_batches(X_train, y_train, args.batch_size):
            yb_oh = to_onehot(yb, n_classes)

            if isinstance(opt, NAG):
                opt.apply_lookahead(model.get_params())

            logits = model.forward(xb)
            loss = loss_fn(logits, yb_oh)
            dloss = grad_fn(logits, yb_oh)

            if isinstance(opt, NAG):
                opt.restore_weights(model.get_params())

            grads = model.backward(dloss)
            opt.update([layer.get_params() for layer in reversed(model.layers)], grads)
            batch_losses.append(loss)

        avg_loss = float(np.mean(batch_losses))
        val_preds = model.predict(X_val)
        val_m = compute_metrics(y_val, val_preds)
        train_preds = model.predict(X_train)
        train_m = compute_metrics(y_train, train_preds)

        print(
            f"epoch {ep}/{args.epochs}  loss={avg_loss:.4f}  "
            f"train_acc={train_m['accuracy']:.4f}  "
            f"val_acc={val_m['accuracy']:.4f}  val_f1={val_m['f1']:.4f}"
        )

        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": ep,
                    "train_loss": avg_loss,
                    "train_acc": train_m["accuracy"],
                    "val_acc": val_m["accuracy"],
                    "val_f1": val_m["f1"],
                    "val_precision": val_m["precision"],
                    "val_recall": val_m["recall"],
                }
            )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            model.save(args.model_save_path)
            cfg = {
                "dataset": args.dataset,
                "input_size": 784,
                "hidden_sizes": sizes,
                "output_size": n_classes,
                "activation": args.activation,
                "weight_init": args.weight_init,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "loss": args.loss,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "best_val_f1": best_f1,
            }
            with open(args.config_save_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"  → saved best model  (val_f1={best_f1:.4f})")

    print("\nevaluating best model on test set ...")
    model.load(args.model_save_path)
    test_preds = model.predict(X_test)
    test_m = compute_metrics(y_test, test_preds)
    print(
        f"test  acc={test_m['accuracy']:.4f}  precision={test_m['precision']:.4f}  "
        f"recall={test_m['recall']:.4f}  f1={test_m['f1']:.4f}"
    )

    if use_wandb:
        import wandb

        wandb.log(
            {
                "test_acc": test_m["accuracy"],
                "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
                "test_f1": test_m["f1"],
            }
        )
        if run is not None:
            run.finish()

    return test_m


# keep get_args as an alias so any existing notebooks / scripts still work
def get_args():
    return parse_arguments()


if __name__ == "__main__":
    train(parse_arguments())

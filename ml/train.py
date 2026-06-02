#!/usr/bin/env python3
"""Train the neural surrogate pricer.

Loads the config, generates the labeled dataset, standardizes features on the
training split, trains the MLP with Adam and MSE loss, tracks validation error
each epoch, and saves the best checkpoint plus a training curve and a metrics
file. This is the training stage of the model lifecycle. Evaluation lives in
evaluate.py and inference in infer.py.

Usage:
    python3 ml/train.py                       # uses ml/configs/default.yaml
    python3 ml/train.py --config my.yaml
    python3 ml/train.py --epochs 20           # override any train field
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

import dataset as data_mod
from model import build_model

HERE = os.path.dirname(__file__)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def to_tensor(a, device):
    return torch.as_tensor(a, device=device)


def evaluate_loss(model, X, y, device, batch=16384):
    model.eval()
    se = 0.0
    n = len(X)
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = to_tensor(X[i:i + batch], device)
            yb = to_tensor(y[i:i + batch], device)
            pred = model(xb)
            se += torch.sum((pred - yb) ** 2).item()
    return se / n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "default.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["train"]["device"] = args.device

    device = pick_device(cfg["train"]["device"])
    torch.manual_seed(cfg["data"]["seed"])
    print(f"Training surrogate pricer on {device}")

    # Data.
    ds = data_mod.generate(cfg["data"])
    train, val, test = data_mod.split(ds, cfg["data"]["val_fraction"],
                                      cfg["data"]["test_fraction"],
                                      cfg["data"]["seed"])
    mean, std = data_mod.standardizer(train.X)
    print(f"  train {len(train.X):,}  val {len(val.X):,}  test {len(test.X):,}")

    # Model.
    model = build_model(cfg["model"], in_features=train.X.shape[1]).to(device)
    model.set_standardizer(mean, std)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.MSELoss()

    Xtr = to_tensor(train.X, device)
    ytr = to_tensor(train.y, device)
    n = len(Xtr)
    batch = cfg["train"]["batch_size"]
    epochs = cfg["train"]["epochs"]

    history = {"train_rmse": [], "val_rmse": []}
    best_val = float("inf")
    ckpt_path = os.path.join(HERE, "..", cfg["train"]["checkpoint"])
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        train_mse = evaluate_loss(model, train.X, train.y, device)
        val_mse = evaluate_loss(model, val.X, val.y, device)
        train_rmse = float(np.sqrt(train_mse))
        val_rmse = float(np.sqrt(val_mse))
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({"state_dict": model.state_dict(),
                        "config": cfg,
                        "feature_names": data_mod.FEATURE_NAMES}, ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_rmse {train_rmse:.5f}  "
                  f"val_rmse {val_rmse:.5f}")

    elapsed = time.perf_counter() - t0
    print(f"\nBest val RMSE {best_val:.5f} per unit strike. "
          f"Trained {epochs} epochs in {elapsed:.1f}s. Saved {ckpt_path}")

    # Persist the training curve and metrics for the report.
    metrics = {
        "device": str(device),
        "epochs": epochs,
        "best_val_rmse": best_val,
        "train_seconds": elapsed,
        "n_train": len(train.X),
        "n_val": len(val.X),
        "n_test": len(test.X),
        "history": history,
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_curve(history, os.path.join(HERE, "loss_curve.png"))
    print("Wrote ml/metrics.json and ml/loss_curve.png")


def _plot_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_rmse"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["train_rmse"], color="#2a78d6", linewidth=2, label="train")
    ax.plot(epochs, history["val_rmse"], color="#008300", linewidth=2, label="validation")
    ax.set_yscale("log")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, color="#e7e7e4", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE per unit strike")
    ax.set_title("Surrogate pricer training", loc="left", fontsize=13)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()

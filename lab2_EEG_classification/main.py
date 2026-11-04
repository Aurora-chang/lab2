"""Configurable EEGNet training. Run from any working directory."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab2-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataloader import prepare_data
from models.EEGNet import EEGNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["epoch", "train_loss", "train_accuracy", "validation_loss",
          "validation_accuracy", "test_loss", "test_accuracy", "learning_rate"]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(args):
    config = json.loads(args.config.read_text())
    for key, target in (("epochs", "epochs"), ("batch_size", "batch_size"), ("lr", "learning_rate")):
        value = getattr(args, key)
        if value is not None:
            config["training"][target] = value
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    config["data_dir"] = str(resolve_path(config["data_dir"]))
    config["output_dir"] = str(resolve_path(config["output_dir"]))
    config["evaluation"].setdefault("split_seed", 42)
    config["model"].setdefault("negative_slope", 0.01)
    config["model"].setdefault("temporal_block", "slide")
    training, evaluation = config["training"], config["evaluation"]
    if config["model"]["name"] != "EEGNet":
        raise ValueError("Only EEGNet is implemented; DeepConvNet is a later bonus")
    if training["optimizer"] != "adam" or training["loss"] != "cross_entropy":
        raise ValueError("Supported optimizer/loss: adam/cross_entropy")
    for key in ("epochs", "batch_size"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "weight_decay"):
        if not math.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if training["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    if not 0 < evaluation["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if evaluation["checkpoint_metric"] != "validation_accuracy":
        raise ValueError("Checkpoints must be selected using validation_accuracy")
    if type(evaluation["stratify"]) is not bool or type(evaluation["record_test_curves"]) is not bool:
        raise ValueError("stratify and record_test_curves must be booleans")
    for seed in (config["seed"], evaluation["split_seed"]):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seeds must be integers in [0, 2**32)")
    name = config["experiment_name"]
    if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("experiment_name must contain only letters, numbers, underscores, or hyphens")
    return config


def epoch_metrics(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered")
            if training:
                loss.backward()
                optimizer.step()
            size = labels.size(0)
            total_loss += loss.item() * size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += size
    return {"loss": total_loss / count, "accuracy": 100.0 * correct / count}


def plot_curves(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [r["epoch"] for r in rows]
    for split in ("train", "validation", "test"):
        if rows[0][f"{split}_loss"] is None:
            continue
        for ax, metric in zip(axes, ("loss", "accuracy")):
            ax.plot(epochs, [r[f"{split}_{metric}"] for r in rows], label=split,
                    marker="o" if len(rows) == 1 else None)
    for ax, title, ylabel in zip(axes, ("Cross-entropy loss", "Accuracy"), ("Loss", "Accuracy (%)")):
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def source_hashes():
    paths = [Path(__file__), PROJECT_ROOT / "lab2_EEG_classification/dataloader.py",
             PROJECT_ROOT / "lab2_EEG_classification/models/EEGNet.py"]
    return {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(config, device, num_threads=None):
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    evaluation = config["evaluation"]
    splits, preprocessing = prepare_data(
        config["data_dir"], evaluation["validation_fraction"],
        evaluation["split_seed"], evaluation["stratify"],
    )
    model_options = {k: v for k, v in config["model"].items() if k != "name"}
    model = EEGNet(**model_options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                         batch_size=config["training"]["batch_size"], shuffle=name == "train",
                         generator=generator if name == "train" else torch.Generator().manual_seed(seed + 1),
                         num_workers=0, pin_memory=device.type == "cuda")
        for name, (x, y) in splits.items()
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(config["output_dir"]) / f"{config['experiment_name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"lab2.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.FileHandler(run_dir / "training.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    try:
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "preprocessing.json", preprocessing)
        dataset_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(config["data_dir"]).glob("*.npz"))}
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "threads": torch.get_num_threads(),
            "source_sha256": source_hashes(), "dataset_sha256": dataset_hashes,
        })
        (run_dir / "model.txt").write_text(str(model) + "\n")
        logger.info("Results: %s", run_dir)
        logger.info("Device: %s; split sizes: %s; parameters: %s", device,
                    preprocessing["sizes"], sum(p.numel() for p in model.parameters()))
        rows, best_accuracy, best_epoch = [], -1.0, None
        peak_test_accuracy, peak_test_epoch = None, None
        started = time.perf_counter()
        with (run_dir / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for epoch in range(1, config["training"]["epochs"] + 1):
                row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
                for split in ("train", "validation", "test"):
                    metrics = (epoch_metrics(model, loaders[split], criterion, device,
                                            optimizer if split == "train" else None)
                               if split != "test" or evaluation["record_test_curves"]
                               else {"loss": None, "accuracy": None})
                    row.update({f"{split}_{key}": value for key, value in metrics.items()})
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                if row["test_accuracy"] is not None and (peak_test_accuracy is None or row["test_accuracy"] > peak_test_accuracy):
                    peak_test_accuracy, peak_test_epoch = row["test_accuracy"], epoch
                improved = row["validation_accuracy"] > best_accuracy
                if improved:
                    best_accuracy, best_epoch = row["validation_accuracy"], epoch
                checkpoint = {
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "config": config,
                    "preprocessing": preprocessing, "metrics": row,
                    "best_validation_accuracy": best_accuracy, "best_epoch": best_epoch,
                }
                # Serialize immediately: later optimizer steps cannot mutate saved weights.
                if improved:
                    save_checkpoint(run_dir / "best.pt", checkpoint)
                save_checkpoint(run_dir / "last.pt", checkpoint)
                plot_curves(rows, run_dir / "curves.png")
                logger.info("Epoch %d/%d | loss %.5f | train %.2f%% | validation %.2f%% | test %s",
                            epoch, config["training"]["epochs"], row["train_loss"],
                            row["train_accuracy"], row["validation_accuracy"],
                            "disabled" if row["test_accuracy"] is None else f"{row['test_accuracy']:.2f}%")
        best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best["model_state_dict"])
        selected_scores = {name: epoch_metrics(model, loader, criterion, device)
                           for name, loader in loaders.items()}
        summary = {
            "status": "complete", "selection_criterion": "validation_accuracy",
            "selected_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
            "selected_checkpoint_scores": selected_scores,
            "peak_test_accuracy": peak_test_accuracy, "peak_test_epoch": peak_test_epoch,
            "peak_test_note": "Maximum over monitored epochs; not an independent held-out estimate.",
            "accuracy_unit": "percent", "duration_seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        write_json(run_dir / "summary.json", summary)
        logger.info("Selected epoch %d; selected checkpoint test accuracy: %.2f%%", best_epoch,
                    selected_scores["test"]["accuracy"])
        if peak_test_accuracy is not None:
            logger.info("Highest monitored testing accuracy: %.2f%% at epoch %d (test-set maximum)",
                        peak_test_accuracy, peak_test_epoch)
        return run_dir
    except BaseException:
        logger.exception("Run did not complete; inspect saved per-epoch artifacts.")
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", "-num_epochs", type=int)
    parser.add_argument("--batch-size", "-batch_size", type=int)
    parser.add_argument("--lr", "-lr", type=float)
    parser.add_argument("--seed", type=int, help="Model seed; does not change the validation split")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-threads", type=int)
    args = parser.parse_args()
    if args.num_threads is not None and args.num_threads < 1:
        parser.error("--num-threads must be positive")
    config = load_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")
    run(config, device, args.num_threads)


if __name__ == "__main__":
    main()

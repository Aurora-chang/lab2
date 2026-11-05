"""Verify a completed CPU/GPU run, including checkpoint inference on its device."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab2_EEG_classification"))
from dataloader import prepare_data
from main import epoch_metrics, source_hashes
from models.EEGNet import EEGNet


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_run(run_dir, data_dir=None, device="cpu", require_cuda=False):
    run_dir = Path(run_dir)
    (run_dir / "verification.json").unlink(missing_ok=True)
    for name in ("config.json", "environment.json", "preprocessing.json", "model.txt",
                 "metrics.csv", "best.pt", "last.pt", "curves.png", "training.log", "summary.json"):
        require((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0,
                f"Missing or empty artifact: {name}")
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    saved_prep = json.loads((run_dir / "preprocessing.json").read_text())
    require(summary["status"] == "complete", "Run did not complete")
    is_bonus = config["model"]["name"] == "DeepConvNet"
    if is_bonus:
        from bonus_training import source_hashes as bonus_source_hashes
        from models.DeepConvNet import DeepConvNet
        expected_hashes = bonus_source_hashes()
        model_class = DeepConvNet
    else:
        require(config["model"]["name"] == "EEGNet", "Unknown model")
        expected_hashes = source_hashes()
        model_class = EEGNet
    require(environment["source_sha256"] == expected_hashes, "Code differs from training source")
    device = torch.device(device)
    if require_cuda:
        require(environment["device"].startswith("cuda") and environment["gpu"],
                "Training was not recorded on a CUDA GPU")
        require(device.type == "cuda" and torch.cuda.is_available(), "GPU checkpoint verification requires CUDA")
    if device.type == "cpu":
        torch.set_num_threads(2)
    data_dir = Path(data_dir or config["data_dir"])
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_dir.glob("*.npz"))}
    require(hashes == environment["dataset_sha256"], "Dataset differs from training data")
    evaluation = config["evaluation"]
    splits, prep = prepare_data(data_dir, evaluation["validation_fraction"],
                               evaluation["split_seed"], evaluation["stratify"])
    require(prep == saved_prep, "Preprocessing or split indices differ")
    with (run_dir / "metrics.csv").open() as stream:
        rows = [{k: float(v) if v else None for k, v in row.items()}
                for row in csv.DictReader(stream)]
    require(len(rows) == config["training"]["epochs"], "Incorrect number of metric rows")
    require([r["epoch"] for r in rows] == list(range(1, len(rows) + 1)), "Invalid epoch sequence")
    for row in rows:
        for split in ("train", "validation", "test"):
            loss, accuracy = row[f"{split}_loss"], row[f"{split}_accuracy"]
            if split == "test" and not evaluation["record_test_curves"]:
                require(loss is None and accuracy is None, "Test curves should be disabled")
                continue
            require(loss is not None and np.isfinite(loss) and loss >= 0, "Invalid loss")
            require(accuracy is not None and np.isfinite(accuracy) and 0 <= accuracy <= 100,
                    "Invalid accuracy")
    selected = max(rows, key=lambda r: r["validation_accuracy"])
    require(summary["selected_epoch"] == selected["epoch"], "Checkpoint was not validation-selected")
    require(summary["best_validation_accuracy"] == selected["validation_accuracy"], "Wrong best validation score")
    if evaluation["record_test_curves"]:
        peak = max(rows, key=lambda r: r["test_accuracy"])
        require(summary["peak_test_epoch"] == peak["epoch"] and
                summary["peak_test_accuracy"] == peak["test_accuracy"], "Incorrect peak test summary")
    else:
        require(summary["peak_test_epoch"] is None and summary["peak_test_accuracy"] is None,
                "Disabled monitoring should not report a peak")
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    require(best["epoch"] == summary["selected_epoch"], "Wrong best checkpoint epoch")
    require(last["epoch"] == len(rows), "Wrong last checkpoint epoch")
    require(bool(best["optimizer_state_dict"]["state"]), "Checkpoint lacks trained optimizer state")
    model = model_class(**{k: v for k, v in config["model"].items() if k != "name"}).to(device)
    model.load_state_dict(best["model_state_dict"])
    matrices = json.loads((run_dir / "confusion_matrices.json").read_text()) if is_bonus else None
    if is_bonus:
        require(matrices["selected_epoch"] == summary["selected_epoch"], "Confusion checkpoint differs")
        for saved in (best, last):
            for key, weights in saved["model_state_dict"].items():
                if key.endswith("weight") and weights.ndim >= 2:
                    limit = config["model"]["classifier_max_norm"] if key.startswith("classify.") else config["model"]["conv_max_norm"]
                    require(bool((weights.flatten(1).norm(2, dim=1) <= limit + 1e-5).all()), "Max norm constraint violated")
    for split in (("train", "validation", "test") if is_bonus else ("validation", "test")):
        x, y = splits[split]
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                            batch_size=config["training"]["batch_size"])
        actual = epoch_metrics(model, loader, torch.nn.CrossEntropyLoss(), device)
        expected = summary["selected_checkpoint_scores"][split]
        require(np.isclose(actual["loss"], expected["loss"], atol=1e-5, rtol=1e-4),
                f"Reloaded {split} loss differs")
        require(np.isclose(actual["accuracy"], expected["accuracy"], atol=1e-6, rtol=0),
                f"Reloaded {split} accuracy differs")
        if is_bonus:
            predicted = []
            model.eval()
            with torch.inference_mode():
                for inputs, _ in loader:
                    predicted.extend(model(inputs.to(device)).argmax(1).cpu().tolist())
            matrix = np.zeros((2, 2), dtype=np.int64)
            np.add.at(matrix, (y, np.asarray(predicted)), 1)
            require(matrix.tolist() == matrices["splits"][split], "Confusion counts differ")
        if split == "validation":
            require(np.isclose(actual["accuracy"], selected["validation_accuracy"], atol=1e-6, rtol=0),
                    "Best checkpoint does not reproduce its validation epoch")
    require((run_dir / "curves.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "Invalid PNG")
    result = {"status": "passed", "run_dir": str(run_dir.resolve()), "verification_device": str(device),
              "training_device": environment["device"], "epochs": len(rows),
              "selected_epoch": summary["selected_epoch"],
              "checks": ["artifacts", "source_and_data_hashes", "preprocessing", "metrics",
                         "validation_selection", "peak_test_summary", "checkpoint_reload", "plot"]}
    if is_bonus:
        result["checks"].extend(["all_split_confusion_matrices", "max_norm_constraints"])
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    # Preserve original GPU verification; direct CLI writes a separate local review.
    original_path = args.run_dir / "verification.json"
    original = original_path.read_bytes() if original_path.exists() else None
    try:
        result = verify_run(args.run_dir, args.data_dir, args.device, args.require_cuda)
        (args.run_dir / "local_review.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if original is not None:
            original_path.write_bytes(original)


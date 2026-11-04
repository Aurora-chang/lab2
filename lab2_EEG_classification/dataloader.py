"""Load EEG data and fit missing-value imputation on training examples only."""
from pathlib import Path
import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def load_split(data_dir, split):
    signals, labels, subjects = [], [], []
    for subject_id, subject in enumerate(("S4b", "X11b")):
        path = Path(data_dir) / f"{subject}_{split}.npz"
        with np.load(path, allow_pickle=False) as data:
            signal = data["signal"]
            label = data["label"]
            if signal.ndim != 3 or signal.shape[1:] != (750, 2):
                raise ValueError(f"{path}: expected signal shape (N, 750, 2)")
            if label.shape != (len(signal),) or not np.isin(label, [1, 2]).all():
                raise ValueError(f"{path}: expected N labels with values 1 or 2")
            if np.isinf(signal).any():
                raise ValueError(f"{path}: signal contains infinite values")
            signals.append(signal.astype(np.float32))
            labels.append(label.astype(np.int64) - 1)
            subjects.append(np.full(len(label), subject_id, dtype=np.int64))
    x = np.concatenate(signals).transpose(0, 2, 1)[:, None, :, :]
    return np.ascontiguousarray(x), np.concatenate(labels), np.concatenate(subjects)


def split_indices(labels, subjects, fraction=0.2, seed=42, stratify=True):
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    groups = subjects * 2 + labels if stratify else np.zeros(len(labels), dtype=int)
    train, validation = [], []
    for group in np.unique(groups):
        indices = rng.permutation(np.flatnonzero(groups == group))
        if len(indices) < 2:
            raise ValueError("Each split group needs at least two examples")
        count = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return np.sort(train), np.sort(validation)


def prepare_data(data_dir=DEFAULT_DATA_DIR, validation_fraction=0.2,
                 split_seed=42, stratify=True):
    x, y, subjects = load_split(data_dir, "train")
    test_x, test_y, _ = load_split(data_dir, "test")
    train_idx, val_idx = split_indices(y, subjects, validation_fraction, split_seed, stratify)
    fill_value = float(np.nanmean(x[train_idx], dtype=np.float64))
    if not np.isfinite(fill_value):
        raise ValueError("Cannot fit imputation: no finite training values")
    splits = {}
    for name, features, labels in (
        ("train", x[train_idx], y[train_idx]),
        ("validation", x[val_idx], y[val_idx]),
        ("test", test_x, test_y),
    ):
        features = np.nan_to_num(features, nan=fill_value)
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite values after preprocessing: {name}")
        splits[name] = (features, labels)
    metadata = {
        "imputation": "scalar_mean_of_training_partition", "fill_value": fill_value,
        "normalization": "none", "split_seed": split_seed,
        "stratification": "subject_and_class" if stratify else "none",
        "train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist(),
        "sizes": {name: len(labels) for name, (_, labels) in splits.items()},
    }
    return splits, metadata


def read_bci_data(data_dir=DEFAULT_DATA_DIR):
    """Compatibility loader without a validation split; uses full train mean."""
    train_x, train_y, _ = load_split(data_dir, "train")
    test_x, test_y, _ = load_split(data_dir, "test")
    fill = float(np.nanmean(train_x, dtype=np.float64))
    if not np.isfinite(fill):
        raise ValueError("Training data has no finite mean")
    return (np.nan_to_num(train_x, nan=fill), train_y,
            np.nan_to_num(test_x, nan=fill), test_y)

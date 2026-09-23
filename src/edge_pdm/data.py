from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import WINDOW_SIZE


def save_dataset(path: Path, windows: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, windows=windows.astype(np.float32), labels=labels)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        dataset = np.load(path)
        return dataset["windows"], dataset["labels"]
    return load_measurement_csv(path)


def load_measurement_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, str], list[tuple[int, float, float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["label"], row["window_id"])
            grouped.setdefault(key, []).append((
                int(row["sample_index"]), float(row["x"]), float(row["y"]), float(row["z"])
            ))
    windows, labels = [], []
    for (label, _window_id), rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        if len(rows) != WINDOW_SIZE:
            raise ValueError(f"window has {len(rows)} samples, expected {WINDOW_SIZE}")
        windows.append([[x, y, z] for _, x, y, z in rows])
        labels.append(label)
    return np.asarray(windows, dtype=np.float32), np.asarray(labels)


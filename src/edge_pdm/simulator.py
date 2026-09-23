from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import CLASSES, SAMPLE_RATE_HZ, WINDOW_SIZE


@dataclass
class MachineSignalGenerator:
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def window(self, label: str) -> np.ndarray:
        if label not in CLASSES:
            raise ValueError(f"unknown class: {label}")

        t = np.arange(WINDOW_SIZE) / SAMPLE_RATE_HZ
        rotation_hz = self.rng.uniform(28, 34)
        phase = self.rng.uniform(0, 2 * np.pi, size=3)
        amplitude = self.rng.uniform(0.8, 1.2)
        signal = np.column_stack([
            0.10 * np.sin(2 * np.pi * rotation_hz * t + phase[0]),
            0.08 * np.sin(2 * np.pi * rotation_hz * t + phase[1]),
            0.05 * np.sin(2 * np.pi * rotation_hz * t + phase[2]),
        ])
        signal += self.rng.normal(0, 0.018, signal.shape)

        if label == "imbalance":
            fundamental = amplitude * 0.55 * np.sin(2 * np.pi * rotation_hz * t)
            signal[:, 0] += fundamental
            signal[:, 1] += 0.35 * fundamental
        elif label == "misalignment":
            harmonic = amplitude * 0.38 * np.sin(2 * np.pi * 2 * rotation_hz * t)
            signal[:, 1] += harmonic
            signal[:, 2] += 0.55 * np.sin(2 * np.pi * rotation_hz * t)
        elif label == "bearing_fault":
            impact_rate = self.rng.uniform(85, 115)
            impacts = np.zeros(WINDOW_SIZE)
            period = max(1, int(SAMPLE_RATE_HZ / impact_rate))
            offset = int(self.rng.integers(0, period))
            impacts[offset::period] = self.rng.uniform(0.7, 1.1)
            resonance = np.exp(-np.arange(30) / 6) * np.sin(2 * np.pi * 320 * np.arange(30) / SAMPLE_RATE_HZ)
            fault = np.convolve(impacts, resonance, mode="full")[:WINDOW_SIZE]
            signal[:, 0] += fault
            signal[:, 2] += 0.45 * fault

        return signal.astype(np.float32)

    def dataset(self, samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        windows, labels = [], []
        for label in CLASSES:
            for _ in range(samples_per_class):
                windows.append(self.window(label))
                labels.append(label)
        indices = self.rng.permutation(len(labels))
        return np.asarray(windows)[indices], np.asarray(labels)[indices]


def scenario_stream(generator: MachineSignalGenerator, dwell: int = 20) -> Iterator[tuple[str, np.ndarray]]:
    index = 0
    while True:
        label = CLASSES[(index // dwell) % len(CLASSES)]
        yield label, generator.window(label)
        index += 1


def post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3) as response:
        response.read()


def run_telemetry_simulator(url: str, interval: float = 0.5) -> None:
    from .model import load_bundle, predict_window

    bundle = load_bundle()
    generator = MachineSignalGenerator()
    for actual, window in scenario_stream(generator):
        started = time.perf_counter()
        prediction, confidence, probabilities, features = predict_window(bundle, window)
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = {
            "device_id": "simulator-01",
            "timestamp": time.time(),
            "prediction": prediction,
            "actual": actual,
            "confidence": confidence,
            "probabilities": probabilities,
            "rms_g": float(np.sqrt(np.mean(window ** 2))),
            "temperature_c": 34.0 + float(generator.rng.normal(0, 0.25)),
            "inference_ms": elapsed_ms,
            "features": features.tolist(),
        }
        try:
            post_json(url, payload)
            print(json.dumps(payload))
        except Exception as exc:
            print(f"telemetry error: {exc}")
        time.sleep(interval)


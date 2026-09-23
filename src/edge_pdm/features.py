import numpy as np

from .config import FEATURE_NAMES, SAMPLE_RATE_HZ, WINDOW_SIZE


def _kurtosis(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    variance = np.mean(centered ** 2)
    if variance < 1e-12:
        return 0.0
    return float(np.mean(centered ** 4) / (variance ** 2))


def extract_features(window: np.ndarray, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Convert one Nx3 acceleration window into firmware-compatible features."""
    values = np.asarray(window, dtype=np.float64)
    if values.shape != (WINDOW_SIZE, 3):
        raise ValueError(f"expected {(WINDOW_SIZE, 3)}, got {values.shape}")

    values = values - np.mean(values, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(values ** 2, axis=0))
    peaks = np.max(np.abs(values), axis=0)
    crest = peaks / np.maximum(rms, 1e-9)
    kurtosis = np.array([_kurtosis(values[:, axis]) for axis in range(3)])

    magnitude = np.linalg.norm(values, axis=1)
    magnitude -= np.mean(magnitude)
    spectrum = np.abs(np.fft.rfft(magnitude)) ** 2
    frequencies = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / sample_rate)
    total_power = max(float(np.sum(spectrum[1:])), 1e-12)
    bands = []
    for low, high in ((0, 40), (40, 120), (120, 300), (300, 501)):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(float(np.sum(spectrum[mask]) / total_power))
    dominant = float(frequencies[1 + np.argmax(spectrum[1:])]) / (sample_rate / 2)
    axis_ratio = float(np.max(rms) / max(np.sum(rms), 1e-9))

    result = np.concatenate([rms, peaks, crest, kurtosis, bands, [dominant, axis_ratio]])
    if len(result) != len(FEATURE_NAMES):
        raise RuntimeError("feature schema mismatch")
    return result.astype(np.float32)


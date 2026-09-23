from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
FIRMWARE_MODEL_HEADER = ROOT / "firmware" / "include" / "model_data.h"

SAMPLE_RATE_HZ = 1000
WINDOW_SIZE = 256
CLASSES = ("normal", "imbalance", "misalignment", "bearing_fault")
FEATURE_NAMES = (
    "rms_x", "rms_y", "rms_z", "peak_x", "peak_y", "peak_z",
    "crest_x", "crest_y", "crest_z", "kurtosis_x", "kurtosis_y",
    "kurtosis_z", "band_0_40", "band_40_120", "band_120_300",
    "band_300_500", "dominant_frequency", "axis_ratio",
)


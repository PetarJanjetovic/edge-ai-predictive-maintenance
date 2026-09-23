import numpy as np
import pytest

from edge_pdm.config import FEATURE_NAMES, WINDOW_SIZE
from edge_pdm.features import extract_features
from edge_pdm.simulator import MachineSignalGenerator, Mpu6050Emulator


def test_feature_vector_is_finite_and_has_stable_schema():
    window = MachineSignalGenerator().window("normal")
    features = extract_features(window)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_feature_extractor_rejects_wrong_window_size():
    with pytest.raises(ValueError, match="expected"):
        extract_features(np.zeros((WINDOW_SIZE - 1, 3)))


def test_fault_signatures_are_distinct():
    generator = MachineSignalGenerator(seed=7)
    normal = extract_features(generator.window("normal"))
    imbalance = extract_features(generator.window("imbalance"))
    bearing = extract_features(generator.window("bearing_fault"))
    assert imbalance[0] > normal[0] * 2
    assert bearing[15] > normal[15]


def test_mpu6050_emulator_quantizes_and_clips_measurements():
    ideal = np.full((WINDOW_SIZE, 3), [20.0, 0.12345, -20.0], dtype=np.float32)
    measured = Mpu6050Emulator(seed=2, noise_std_g=0, bias_std_g=0).sample(ideal)
    assert measured.max() <= 8.0
    assert measured.min() >= -8.0
    assert np.allclose(measured * 4096, np.rint(measured * 4096))

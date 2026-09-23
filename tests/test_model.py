import numpy as np

from edge_pdm.config import CLASSES
from edge_pdm.model import predict_window, train
from edge_pdm.simulator import MachineSignalGenerator


def test_small_dataset_trains_and_predicts():
    generator = MachineSignalGenerator(seed=11)
    windows, labels = generator.dataset(35)
    model, metrics = train(windows, labels, seed=11)
    assert metrics["accuracy"] >= 0.75
    prediction, confidence, probabilities, features = predict_window(
        model, generator.window("imbalance")
    )
    assert prediction in CLASSES
    assert 0 <= confidence <= 1
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert len(features) == 18


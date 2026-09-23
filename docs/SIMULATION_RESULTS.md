# Virtual Hardware Simulation Results

Run date: September 23, 2026

## Configuration

- Virtual controller: ESP32-S3
- Virtual sensor: MPU6050 at +/-8 g
- Sensor conversion: 4096 LSB/g
- Added effects: per-axis bias, white noise, cross-axis coupling, clipping, and quantization
- Evaluation set: 2,000 windows, with 500 windows per class
- Random seed: 2026

## Results

| Metric | Result |
|---|---:|
| Simulated accuracy | 100.00% |
| Simulated macro F1 | 1.0000 |
| Mean confidence | 99.72% |
| Estimated inference latency, mean | 2.20 ms |
| Estimated inference latency, p95 | 2.49 ms |
| Compiled RAM use | 22,372 bytes |
| Compiled flash use | 316,493 bytes |

## Confusion matrix

| Actual / predicted | Normal | Imbalance | Misalignment | Bearing fault |
|---|---:|---:|---:|---:|
| Normal | 500 | 0 | 0 | 0 |
| Imbalance | 0 | 500 | 0 | 0 |
| Misalignment | 0 | 0 | 500 | 0 |
| Bearing fault | 0 | 0 | 0 | 500 |

## Interpretation

This run verifies that sensor imperfections do not break the software, DSP, model, telemetry, or firmware assumptions. The perfect result also shows that the generated fault signatures are easier to separate than faults recorded from physical machinery.

Accuracy, confidence, and latency in this document are simulated or estimated unless explicitly labeled as compiled memory usage. They must not be presented as physical test results. Resume claims should use measurements collected after deploying to an ESP32-S3 and MPU6050.

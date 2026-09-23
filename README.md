# Edge AI Predictive Maintenance System

An end-to-end embedded machine-learning system that identifies mechanical faults from vibration data. An ESP32-S3 samples an MPU6050 accelerometer, extracts signal features, and runs an offline classifier. A local dashboard displays live health, confidence, vibration severity, and inference latency.

The repository also includes a deterministic mechanical-signal simulator, so the complete data-to-dashboard pipeline works before hardware arrives.

## System architecture

```text
MPU6050 (1 kHz) -> ESP32-S3 -> windowing + DSP -> neural classifier
                                             |-> LED/buzzer alert
                                             `-> JSON telemetry -> dashboard

Synthetic generator -> labeled windows -> training/evaluation -> C++ model header
```

## Fault classes

| Class | Simulated signature | Physical interpretation |
|---|---|---|
| `normal` | Stable fundamental and low noise | Healthy rotating equipment |
| `imbalance` | Elevated 1x rotational component | Uneven mass distribution |
| `misalignment` | Strong 2x component and axial vibration | Shaft/coupling misalignment |
| `bearing_fault` | Repetitive high-frequency impacts | Localized bearing damage |

## Quick start without hardware

```bash
make setup
make data
make train
make dashboard
```

In a second terminal:

```bash
make simulate
```

Open `http://127.0.0.1:8000`. The simulator rotates through healthy and faulty machine states while using the trained model for predictions.

## Hardware

- ESP32-S3 development board
- MPU6050 accelerometer/gyroscope module
- Optional active buzzer and status LED
- USB cable and jumper wires
- Small fan or DC motor for data collection

### Wiring

| MPU6050 | ESP32-S3 |
|---|---|
| VCC | 3.3 V |
| GND | GND |
| SDA | GPIO 8 |
| SCL | GPIO 9 |
| INT | Not required |

The firmware uses 1 kHz sampling and 256-sample windows. Change pins and thresholds in `firmware/include/config.h` for a different board.

## Commands

```bash
edge-pdm generate --samples-per-class 500
edge-pdm train
edge-pdm evaluate
edge-pdm dashboard --port 8000
edge-pdm simulate --url http://127.0.0.1:8000/api/telemetry
edge-pdm board-benchmark --samples-per-class 250
edge-pdm serial --port /dev/cu.usbmodemXXXX
```

## Training on real data

Place each captured window in `data/raw/measurements.csv` with columns:

```text
label,window_id,sample_index,x,y,z
normal,0,0,...
```

Then run `edge-pdm train --input data/raw/measurements.csv`. Keep the sensor mounting position, sample rate, and motor speed consistent across classes. Collect several recording sessions and split by session, not randomly by individual window, to avoid optimistic results.

## Firmware deployment

Training creates `firmware/include/model_data.h`. Install PlatformIO, connect the ESP32-S3, and run:

```bash
cd firmware
PLATFORMIO_CORE_DIR=../.platformio pio run -t upload
PLATFORMIO_CORE_DIR=../.platformio pio device monitor -b 115200
```

The firmware emits one JSON object per inference window. Use the serial bridge to forward it to the dashboard.

## Evaluation targets

- Macro F1 score above 0.90 on held-out recording sessions
- Inference under 10 ms on ESP32-S3
- Total model and preprocessing memory under 100 KB
- False-alarm rate below 2% during a 30-minute healthy run
- Stable operation for at least one hour

## Virtual hardware mode

The telemetry simulator passes each ideal machine signal through an MPU6050 emulator before inference. It adds fixed sensor bias, white measurement noise, cross-axis coupling, clipping at the configured +/-8 g range, and 4096 LSB/g quantization. Run `edge-pdm board-benchmark` for a repeatable virtual-board report.

These figures estimate expected behavior and are useful for software integration. They are not substitutes for measurements from a physical sensor, motor, or ESP32.

## Responsible claims

Synthetic-data accuracy validates the software pipeline, not real-world fault detection. Resume metrics should come from held-out physical recordings and on-device measurements. The system is an educational prototype, not a safety-certified industrial monitor.

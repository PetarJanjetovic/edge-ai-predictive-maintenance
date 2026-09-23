# Demonstration Plan

## Two-minute walkthrough

1. Show the running dashboard in the healthy state.
2. Attach a small clip-on weight to the fan or motor to create imbalance.
3. Show the on-device prediction, confidence change, and alert output.
4. Disconnect the dashboard network and demonstrate that local inference and the LED/buzzer continue operating.
5. Open the evaluation report and compare latency, memory, accuracy, and false-alarm results.

## Evidence to capture

- Confusion matrix from held-out physical recordings
- Serial output showing measured on-device inference latency
- Short video transitioning between normal and imbalance states
- Photo of the sensor mounting and wiring
- Power and memory measurements from the firmware build
- Thirty-minute healthy run used to calculate false-alarm rate

## Interview discussion points

- Why spectral and statistical features were used instead of raw-signal inference
- How synthetic data accelerated integration but was excluded from final claims
- How model size, latency, and accuracy were traded off
- How sensor mounting and recording-session leakage affect generalization
- How a production version would use session-based validation and calibrated alert thresholds


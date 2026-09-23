#include "features.h"

#include <Arduino.h>
#include <cmath>

#include "config.h"

namespace {
constexpr float kPi = 3.14159265358979323846f;

float band_power(const float* values, size_t count, float low_hz, float high_hz,
                 float* total_power, float* dominant_frequency) {
  float power = 0.0f;
  for (size_t bin = 1; bin <= count / 2; ++bin) {
    const float frequency = bin * config::kSampleRateHz / count;
    float real = 0.0f;
    float imag = 0.0f;
    for (size_t sample = 0; sample < count; ++sample) {
      const float angle = 2.0f * kPi * bin * sample / count;
      real += values[sample] * cosf(angle);
      imag -= values[sample] * sinf(angle);
    }
    const float bin_power = real * real + imag * imag;
    if (bin_power > *total_power) {
      *total_power = bin_power;
      *dominant_frequency = frequency;
    }
    if (frequency >= low_hz && frequency < high_hz) power += bin_power;
  }
  return power;
}
}  // namespace

void extract_features(const float samples[][3], size_t count, float* output) {
  float mean[3] = {};
  for (size_t i = 0; i < count; ++i)
    for (size_t axis = 0; axis < 3; ++axis) mean[axis] += samples[i][axis] / count;

  float rms[3] = {}, peak[3] = {}, fourth[3] = {};
  float magnitude[config::kWindowSize];
  float magnitude_mean = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    float centered[3];
    for (size_t axis = 0; axis < 3; ++axis) {
      centered[axis] = samples[i][axis] - mean[axis];
      const float squared = centered[axis] * centered[axis];
      rms[axis] += squared / count;
      fourth[axis] += squared * squared / count;
      peak[axis] = fmaxf(peak[axis], fabsf(centered[axis]));
    }
    magnitude[i] = sqrtf(centered[0]*centered[0] + centered[1]*centered[1] + centered[2]*centered[2]);
    magnitude_mean += magnitude[i] / count;
  }
  for (size_t i = 0; i < count; ++i) magnitude[i] -= magnitude_mean;
  for (size_t axis = 0; axis < 3; ++axis) rms[axis] = sqrtf(rms[axis]);

  size_t index = 0;
  for (float value : rms) output[index++] = value;
  for (float value : peak) output[index++] = value;
  for (size_t axis = 0; axis < 3; ++axis) output[index++] = peak[axis] / fmaxf(rms[axis], 1e-9f);
  for (size_t axis = 0; axis < 3; ++axis) output[index++] = fourth[axis] / fmaxf(powf(rms[axis], 4), 1e-12f);

  float max_bin_power = 0.0f, dominant = 0.0f;
  float bands[4];
  bands[0] = band_power(magnitude, count, 0, 40, &max_bin_power, &dominant);
  bands[1] = band_power(magnitude, count, 40, 120, &max_bin_power, &dominant);
  bands[2] = band_power(magnitude, count, 120, 300, &max_bin_power, &dominant);
  bands[3] = band_power(magnitude, count, 300, 501, &max_bin_power, &dominant);
  const float total = fmaxf(bands[0] + bands[1] + bands[2] + bands[3], 1e-12f);
  for (float value : bands) output[index++] = value / total;
  output[index++] = dominant / (config::kSampleRateHz / 2.0f);
  output[index++] = fmaxf(rms[0], fmaxf(rms[1], rms[2])) / fmaxf(rms[0] + rms[1] + rms[2], 1e-9f);
}


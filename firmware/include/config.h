#pragma once

#include <Arduino.h>

namespace config {
constexpr uint8_t kSdaPin = 8;
constexpr uint8_t kSclPin = 9;
constexpr uint8_t kStatusLedPin = LED_BUILTIN;
constexpr uint8_t kBuzzerPin = 6;
constexpr size_t kWindowSize = 256;
constexpr float kSampleRateHz = 1000.0f;
constexpr uint32_t kSamplePeriodUs = 1000;
constexpr float kAlertConfidence = 0.75f;
}  // namespace config


#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino.h>
#include <Wire.h>

#include "config.h"
#include "features.h"
#include "inference.h"
#include "model_data.h"

Adafruit_MPU6050 mpu;
float samples[config::kWindowSize][3];

void setup() {
  Serial.begin(115200);
  Wire.begin(config::kSdaPin, config::kSclPin);
  pinMode(config::kStatusLedPin, OUTPUT);
  pinMode(config::kBuzzerPin, OUTPUT);
  if (!mpu.begin()) {
    Serial.println("{\"error\":\"MPU6050 not found\"}");
    while (true) delay(1000);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
}

void loop() {
  float sum_squares = 0.0f;
  for (size_t i = 0; i < config::kWindowSize; ++i) {
    const uint32_t started = micros();
    sensors_event_t acceleration, gyro, temperature;
    mpu.getEvent(&acceleration, &gyro, &temperature);
    samples[i][0] = acceleration.acceleration.x / SENSORS_GRAVITY_STANDARD;
    samples[i][1] = acceleration.acceleration.y / SENSORS_GRAVITY_STANDARD;
    samples[i][2] = acceleration.acceleration.z / SENSORS_GRAVITY_STANDARD;
    sum_squares += samples[i][0]*samples[i][0] + samples[i][1]*samples[i][1] + samples[i][2]*samples[i][2];
    while (micros() - started < config::kSamplePeriodUs) yield();
  }

  float features[model_data::kFeatureCount];
  extract_features(samples, config::kWindowSize, features);
  const uint32_t inference_started = micros();
  const Prediction prediction = run_inference(features);
  const float inference_ms = (micros() - inference_started) / 1000.0f;
  const bool alert = strcmp(model_data::kClassNames[prediction.class_index], "normal") != 0 &&
                     prediction.confidence >= config::kAlertConfidence;
  digitalWrite(config::kStatusLedPin, alert ? HIGH : LOW);
  digitalWrite(config::kBuzzerPin, alert ? HIGH : LOW);

  Serial.printf("{\"device_id\":\"esp32-pdm-01\",\"prediction\":\"%s\",\"confidence\":%.5f,",
                model_data::kClassNames[prediction.class_index], prediction.confidence);
  Serial.print("\"probabilities\":{");
  for (size_t i = 0; i < model_data::kClassCount; ++i) {
    if (i) Serial.print(',');
    Serial.printf("\"%s\":%.5f", model_data::kClassNames[i], prediction.probabilities[i]);
  }
  Serial.printf("},\"rms_g\":%.5f,\"inference_ms\":%.3f,\"features\":[",
                sqrtf(sum_squares / (config::kWindowSize * 3)), inference_ms);
  for (size_t i = 0; i < model_data::kFeatureCount; ++i) {
    if (i) Serial.print(',');
    Serial.print(features[i], 6);
  }
  Serial.println("]}");
}

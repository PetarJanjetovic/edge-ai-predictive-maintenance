#pragma once

#include <stddef.h>

struct Prediction {
  size_t class_index;
  float confidence;
  float probabilities[4];
};

Prediction run_inference(const float* features);


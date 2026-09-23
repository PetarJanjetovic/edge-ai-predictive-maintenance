#include "inference.h"

#include <cmath>

#include "model_data.h"

Prediction run_inference(const float* features) {
  float hidden[model_data::kHiddenCount];
  for (size_t row = 0; row < model_data::kHiddenCount; ++row) {
    float sum = model_data::kHiddenBias[row];
    for (size_t col = 0; col < model_data::kFeatureCount; ++col) {
      const float normalized = (features[col] - model_data::kFeatureMean[col]) /
                               model_data::kFeatureScale[col];
      sum += normalized * model_data::kInputWeights[row * model_data::kFeatureCount + col];
    }
    hidden[row] = sum > 0.0f ? sum : 0.0f;
  }

  float logits[model_data::kClassCount];
  float maximum = -INFINITY;
  for (size_t row = 0; row < model_data::kClassCount; ++row) {
    float sum = model_data::kOutputBias[row];
    for (size_t col = 0; col < model_data::kHiddenCount; ++col) {
      sum += hidden[col] * model_data::kOutputWeights[row * model_data::kHiddenCount + col];
    }
    logits[row] = sum;
    maximum = fmaxf(maximum, sum);
  }

  Prediction result{};
  float denominator = 0.0f;
  for (size_t index = 0; index < model_data::kClassCount; ++index) {
    result.probabilities[index] = expf(logits[index] - maximum);
    denominator += result.probabilities[index];
  }
  for (size_t index = 0; index < model_data::kClassCount; ++index) {
    result.probabilities[index] /= denominator;
    if (result.probabilities[index] > result.confidence) {
      result.confidence = result.probabilities[index];
      result.class_index = index;
    }
  }
  return result;
}


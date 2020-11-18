#ifndef TENSORFLOW_SERVING_MODEL_CONFIG_H
#define TENSORFLOW_SERVING_MODEL_CONFIG_H

#include <string>
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {
struct ModelConfig {
  std::string signature_name;

  int inter_threads;
  int intra_threads;
  bool warmup;
};

class ModelConfigFactory {
 public:
  static Status Create(const char* model_config, ModelConfig** config);
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_CONFIG_H


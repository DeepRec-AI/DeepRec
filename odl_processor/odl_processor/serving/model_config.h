#ifndef TENSORFLOW_SERVING_MODEL_CONFIG_H
#define TENSORFLOW_SERVING_MODEL_CONFIG_H

#include <string>

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
  static ModelConfig* Create(const char* model_config);
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_CONFIG_H


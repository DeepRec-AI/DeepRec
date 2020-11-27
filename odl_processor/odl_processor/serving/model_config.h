#ifndef TENSORFLOW_SERVING_MODEL_CONFIG_H
#define TENSORFLOW_SERVING_MODEL_CONFIG_H

#include <string>
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {
struct ModelConfig {
  // Model Info
  std::string signature_name;

  // Run Info
  int inter_threads;
  int intra_threads;

  // Embedding Config
  bool local_storage;
  std::string redis_url;
  std::string redis_password;

  // OSS Config
  std::string oss_endpoint;
  std::string oss_access_id;
  std::string oss_access_key;
};

class ModelConfigFactory {
 public:
  static Status Create(const char* model_config, ModelConfig** config);
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_CONFIG_H


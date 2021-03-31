#ifndef TENSORFLOW_SERVING_MODEL_CONFIG_H
#define TENSORFLOW_SERVING_MODEL_CONFIG_H

#include <string>
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {
struct ModelConfig {

  // Model Info
  std::string checkpoint_dir;
  std::string savedmodel_dir;
  std::string signature_name;
  std::string serialize_protocol;
  int init_timeout_minutes = 0;

  // Run Info
  int inter_threads = 1;
  int intra_threads = 1;

  // Embedding Config
  std::string feature_store_type;
  std::string redis_url;
  std::string redis_password;
  // default db = 0
  size_t redis_db_idx = 0;
  // default 9000s
  int lock_timeout = 15 * 60;
  int read_thread_num = 1;
  int update_thread_num = 1;

  // OSS Config
  std::string model_store_type;
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


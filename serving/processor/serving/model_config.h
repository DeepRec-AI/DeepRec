#ifndef SERVING_PROCESSOR_SERVING_MODEL_CONFIG_H
#define SERVING_PROCESSOR_SERVING_MODEL_CONFIG_H

#include <string>
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {
struct ModelConfig {

  // Model Info
  bool enable_incr_model_update = true;
  std::string checkpoint_dir;
  std::string savedmodel_dir;
  std::string signature_name;
  std::string warmup_file_name;
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
  std::string oss_endpoint = "";
  std::string oss_access_id = "";
  std::string oss_access_key = "";

  // shard user embedding
  std::string id_type = "";
  int shard_instance_count = 0;
  bool shard_embedding = false;
  std::vector<std::string> shard_embedding_names;

  // session num of session group,
  // default num is 1
  int session_num = 1;
  // In multi-session mode, we have two policy for
  // select session for each thread.
  // "RR": Round-Robin policy, threads will use all sessions in Round-Robin way
  // "MOD": Thread select session according unique id, uid % session_num
  std::string select_session_policy = "MOD";

  // session use self-owned thread pool
  bool use_per_session_threads = false;

  // EmbeddingVariable Config
  embedding::StorageType storage_type = embedding::StorageType::INVALID;
  std::string storage_path;
  std::vector<int64> storage_size;
};

class ModelConfigFactory {
 public:
  static Status Create(const char* model_config, ModelConfig** config);
};

} // processor
} // tensorflow

#endif // SERVING_PROCESSOR_SERVING_MODEL_CONFIG_H


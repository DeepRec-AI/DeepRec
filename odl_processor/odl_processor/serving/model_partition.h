#ifndef ODL_PROCESSOR_SERVING_MODEL_PARTITION_H
#define ODL_PROCESSOR_SERVING_MODEL_PARTITION_H

#include "odl_processor/serving/model_config.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace processor {

class PartitionPolicy {
 public:
  static PartitionPolicy* GetGlobalPolicy() {
    static PartitionPolicy pp;
    return &pp;
  }

  void Init(ModelConfig* config) {
    if (!config->shard_embedding) return;

    // get service name
    // serice/service-1/service-2
    Status s = ReadStringFromEnvVar("SERVICE_NAME", "", &instance_name_);
    if (!s.ok() || instance_name_.empty()) {
      LOG(FATAL) << "Get instance SERVICE_NAME failed, " << s.error_message();
    }

    ParsePodName();
    shard_instance_count_ = config->shard_instance_count;
    embedding_group_id_ = instance_id_ % shard_instance_count_;
    LOG(INFO) << "instance_id_: " << instance_id_
              << ", shard_instance_count_: " << shard_instance_count_
              << ", embedding_group_id_: " << embedding_group_id_;
  }

  int GetEmbeddingGroupId() const {
    return embedding_group_id_;
  }

  int GetShardInstanceCount() const {
    return shard_instance_count_;
  }

  int GetInstanceId() const {
    return instance_id_;
  }

 private:
  // service name: serice/service-1/service-2
  void ParsePodName() {
    int idx = instance_name_.length() - 1;
    std::string id_str("");
    while (idx >= 0 && instance_name_[idx] != '_') {
      if (!(instance_name_[idx] >= '0' && instance_name_[idx] <= '9')) break;
      id_str = instance_name_[idx--] + id_str;
    }
    instance_id_ = 0;
    // must match pattern: xxx_123
    if (idx >= 0 && instance_name_[idx] == '_') {
      instance_id_ = std::stoi(id_str);
    }
  }

 private:
  int instance_id_;
  int shard_instance_count_;
  int embedding_group_id_;
  std::string instance_name_ = "";
};

} // namespace processor
} // namespace processor

#endif // ODL_PROCESSOR_SERVING_MODEL_PARTITION_H

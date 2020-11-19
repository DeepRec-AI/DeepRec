#include "odl_processor/serving/model_config.h"
#include "include/json/json.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/cc/saved_model/signature_constants.h"

namespace tensorflow {
namespace processor {

namespace {
constexpr int DEFAULT_CPUS = 8;
}

Status ModelConfigFactory::Create(const char* model_config, ModelConfig** config) {
  if (strlen(model_config) <= 0) {
    return Status(error::Code::INVALID_ARGUMENT,
        "[TensorFlow] Invalid ModelConfig json.");
  }

  Json::Reader reader;
  Json::Value json_config;
  if (!reader.parse(model_config, json_config)) {
    return Status(error::Code::INVALID_ARGUMENT,
        "[TensorFlow] Parse ModelConfig json failed.");
  }

  int64 schedule_threads;
  ReadInt64FromEnvVar("SCHEDULABLE_CPUS", DEFAULT_CPUS, &schedule_threads);

  *config = new ModelConfig;
  if (!json_config["inter_op_parallelism_threads"].isNull()) {
    (*config)->inter_threads =
      json_config["inter_op_parallelism_threads"].asInt();
  } else {
    (*config)->inter_threads = schedule_threads / 2;
  }

  if (!json_config["intra_op_parallelism_threads"].isNull()) {
    (*config)->intra_threads =
      json_config["intra_op_parallelism_threads"].asInt();
  } else {
    (*config)->intra_threads = schedule_threads / 2;
  }

  if (!json_config["signature_name"].isNull()) {
    (*config)->signature_name =
      json_config["signature_name"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No signature_name in ModelConfig.");
  }

  if (!json_config["oss_endpoint"].isNull()) {
    (*config)->oss_endpoint =
      json_config["oss_endpoint"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No oss_endpoint in ModelConfig.");
  }

  if (!json_config["oss_access_id"].isNull()) {
    (*config)->oss_access_id =
      json_config["oss_access_id"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No oss_access_id in ModelConfig.");
  }

  if (!json_config["oss_access_key"].isNull()) {
    (*config)->oss_access_key =
      json_config["oss_access_key"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No oss_access_key in ModelConfig.");
  }

  if (!json_config["enable_warm_up"].isNull()) {
    (*config)->warmup =
      json_config["enable_warm_up"].asBool();
  }

  if (!json_config["local_storage"].isNull()) {
    (*config)->local_storage =
      json_config["local_storage"].asBool();
  }

  return Status::OK();
}

} // processor
} // tensorflow

#include "model_config.h"
#include "include/json/json.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/cc/saved_model/signature_constants.h"

namespace tensorflow {
namespace processor {

namespace {
constexpr int DEFAULT_CPUS = 8;
}

ModelConfig* ModelConfigFactory::Create(const char* model_config) {
  int64 schedule_threads;
  ReadInt64FromEnvVar("SCHEDULABLE_CPUS", DEFAULT_CPUS, &schedule_threads);
  
  ModelConfig* config = new ModelConfig;
  if (strlen(model_config) > 0) {
    Json::Reader reader;
    Json::Value json_config;

    if (!reader.parse(model_config, json_config)) {
      // logging here, show model failure.
      config->inter_threads = schedule_threads / 2;
      config->intra_threads = schedule_threads / 2;
      config->signature_name = kPredictMethodName;
      config->warmup = false; 
    }

    if (!json_config["inter_op_parallelism_threads"].isNull()) {
      config->inter_threads =
        json_config["inter_op_parallelism_threads"].asInt();
    }
    if (!json_config["intra_op_parallelism_threads"].isNull()) {
      config->intra_threads =
        json_config["intra_op_parallelism_threads"].asInt();
    }
    if (!json_config["signature_name"].isNull()) {
      config->signature_name =
        json_config["signature_name"].asString();
    }
    if (!json_config["enable_warm_up"].isNull()) {
      config->warmup =
        json_config["enable_warm_up"].asBool();
    } 
  } else {
    config->inter_threads = schedule_threads / 2;
    config->intra_threads = schedule_threads / 2;
    config->signature_name = kPredictMethodName;
    config->warmup = false;
  }
  return config;
}

} // processor
} // tensorflow

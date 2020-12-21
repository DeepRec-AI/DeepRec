#include "odl_processor/serving/model_config.h"
#include "include/json/json.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace processor {

namespace {
constexpr int DEFAULT_CPUS = 8;
Status AddOSSAccessPrefix(std::string& dir,
                          const ModelConfig* config) {
  auto offset = dir.find("oss://");
  // error oss format
  if (offset == std::string::npos) {
    return tensorflow::errors::Internal(
        "Invalid user input oss dir, ", dir);
  }

  std::string tmp(dir.substr(6));
  offset = tmp.find("/");
  if (offset == std::string::npos) {
    return tensorflow::errors::Internal(
        "Invalid user input oss dir, ", dir);
  }
  
  dir = strings::StrCat(dir.substr(0, offset+6),
                        "\x01id=", config->oss_access_id,
                        "\x02key=", config->oss_access_key,
                        "\x02host=", config->oss_endpoint,
                        tmp.substr(offset));
  return Status::OK();
}
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

  if (!json_config["processor_type"].isNull()) {
    (*config)->processor_type =
      json_config["processor_type"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No processor_type in ModelConfig.");
  }

  if ((*config)->processor_type.empty()) {
    return Status(error::Code::INVALID_ARGUMENT,
        "[TensorFlow] processor_type shouldn't be empty string.");
  }

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

  if (!json_config["init_timeout_minutes"].isNull()) {
    (*config)->init_timeout_minutes =
      json_config["init_timeout_minutes"].asInt();
  } else {
    (*config)->init_timeout_minutes = -1;
  }

  if (!json_config["signature_name"].isNull()) {
    (*config)->signature_name =
      json_config["signature_name"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No signature_name in ModelConfig.");
  }

  if ((*config)->signature_name.empty()) {
    return Status(error::Code::INVALID_ARGUMENT,
        "[TensorFlow] Signature_name shouldn't be empty string.");
  }
  
  if (!json_config["checkpoint_dir"].isNull()) {
    (*config)->checkpoint_dir =
      json_config["checkpoint_dir"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No checkpoint_dir in ModelConfig.");
  }

  if (!json_config["savedmodel_dir"].isNull()) {
    (*config)->savedmodel_dir =
      json_config["savedmodel_dir"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No savedmodel_dir in ModelConfig.");
  }
  
  if ((*config)->processor_type == "odl") {
    if (!json_config["feature_store_type"].isNull()) {
      (*config)->feature_store_type =
        json_config["feature_store_type"].asString();
    } else {
      return Status(error::Code::NOT_FOUND,
          "[TensorFlow] No feature_store_type in ModelConfig.");
    }
  }

  if ((*config)->feature_store_type == "cluster_redis") {
    if (!json_config["redis_url"].isNull()) {
      (*config)->redis_url =
        json_config["redis_url"].asString();
    } else {
      return Status(error::Code::NOT_FOUND,
          "[TensorFlow] Should set redis_url in ModelConfig \
          when feature_store_type=cluster_redis.");
    }

    if (!json_config["redis_password"].isNull()) {
      (*config)->redis_password =
        json_config["redis_password"].asString();
    } else {
      return Status(error::Code::NOT_FOUND,
          "[TensorFlow] Should set redis_password in ModelConfig \
          when feature_store_type=cluster_redis.");
    }
  }
 
  // when feature_store_type == local_redis | cluster_redis
  if ((*config)->feature_store_type.find("redis") != std::string::npos) {
    if (!json_config["read_thread_num"].isNull()) {
      (*config)->read_thread_num =
        json_config["read_thread_num"].asInt();
    } else {
      (*config)->read_thread_num = 4;
    }

    if (!json_config["update_thread_num"].isNull()) {
      (*config)->update_thread_num =
        json_config["update_thread_num"].asInt();
    } else {
      (*config)->update_thread_num = 2;
    }
  }

  if (!json_config["model_store_type"].isNull()) {
    (*config)->model_store_type =
      json_config["model_store_type"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No model_store_type in ModelConfig.");
  }

  if ((*config)->model_store_type != "local") {
    if ((*config)->checkpoint_dir.find((*config)->model_store_type)
        == std::string::npos) {
      return Status(error::Code::INVALID_ARGUMENT,
          "[TensorFlow] Mismatch model_store_type and checkpoint_dir.");
    }

    if ((*config)->savedmodel_dir.find((*config)->model_store_type)
        == std::string::npos) {
      return Status(error::Code::INVALID_ARGUMENT,
          "[TensorFlow] Mismatch model_store_type and savedmodel_dir.");
    }
  }

  if ((*config)->model_store_type == "oss") {
    if (!json_config["oss_endpoint"].isNull()) {
      (*config)->oss_endpoint =
        json_config["oss_endpoint"].asString();
    } else {
      return Status(error::Code::NOT_FOUND,
          "[TensorFlow] No oss_endpoint in ModelConfig.");
    }
  
    if ((*config)->oss_endpoint.empty()) {
      return Status(error::Code::INVALID_ARGUMENT,
          "[TensorFlow] oss_endpoint shouldn't be empty string.");
    }

    if (!json_config["oss_access_id"].isNull()) {
      (*config)->oss_access_id =
        json_config["oss_access_id"].asString();
    } else {
      return Status(error::Code::NOT_FOUND,
          "[TensorFlow] No oss_access_id in ModelConfig.");
    }
    
    if ((*config)->oss_access_id.empty()) {
      return Status(error::Code::INVALID_ARGUMENT,
          "[TensorFlow] oss_access_id shouldn't be empty string.");
    }

    if (!json_config["oss_access_key"].isNull()) {
      (*config)->oss_access_key =
        json_config["oss_access_key"].asString();
    } else {
      return Status(error::Code::NOT_FOUND,
          "[TensorFlow] No oss_access_key in ModelConfig.");
    }
    
    if ((*config)->oss_access_key.empty()) {
      return Status(error::Code::INVALID_ARGUMENT,
          "[TensorFlow] oss_access_key shouldn't be empty string.");
    }

    TF_RETURN_IF_ERROR(AddOSSAccessPrefix(
          (*config)->savedmodel_dir, *config));
    TF_RETURN_IF_ERROR(AddOSSAccessPrefix(
          (*config)->checkpoint_dir, *config));
  }

  return Status::OK();
}

} // processor
} // tensorflow

#include <stdlib.h>

#include "serving/processor/serving/model_config.h"
#include "serving/processor/serving/tracer.h"
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

void ParseGPUIds(const std::string& gpu_ids_list,
                 std::vector<size_t>* gpu_ids) {
  if (!gpu_ids_list.empty()) {
    std::vector<string> ids =
        str_util::Split(gpu_ids_list, ',');
    for (auto id : ids) {
      gpu_ids->emplace_back(std::stoi(id));
    }
  }
}

}

Status ModelConfigFactory::Create(const char* model_config, ModelConfig** config) {
  if (strlen(model_config) <= 0) {
    return Status(error::Code::INVALID_ARGUMENT,
        "[TensorFlow] Invalid ModelConfig json.");
  }

  // Enable INFERENCE_MODE by default
  if (setenv("INFERENCE_MODE", "1", 1) != 0) {
    LOG(WARNING) << "Set env INFERENCE_MODE=1 error.";
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

  // User set session group cpuset,
  // Usage: "0-10;11-20;21-30" or
  //        "0,1,2,3;4,5,6,7;8,9,10"
  if (!json_config["cpusets"].isNull()) {
    (*config)->cpusets =
      json_config["cpusets"].asString();
  }

  if (!json_config["session_num"].isNull()) {
    (*config)->session_num =
      json_config["session_num"].asInt();
  } else {
    (*config)->session_num = 1;
  }

  if (!json_config["gpu_ids_list"].isNull()) {
    std::string gpu_ids_list =
      json_config["gpu_ids_list"].asString();
    ParseGPUIds(gpu_ids_list, &((*config)->gpu_ids));
  }

  bool use_multi_stream = false;
  if (!json_config["use_multi_stream"].isNull()) {
    use_multi_stream = json_config["use_multi_stream"].asBool();
  }
  (*config)->use_multi_stream = use_multi_stream;

  (*config)->select_session_policy = "MOD";
  if (!json_config["select_session_policy"].isNull()) {
    (*config)->select_session_policy =
      json_config["select_session_policy"].asString();
  }
  if ((*config)->select_session_policy != "MOD" &&
      (*config)->select_session_policy != "RR") {
    return Status(error::Code::INVALID_ARGUMENT,
        "[TensorFlow] select_session_policy must be 'RR' or 'MOD'");
  }

  bool enable_inline_execute = false;
  if (!json_config["enable_inline_execute"].isNull()) {
    enable_inline_execute = json_config["enable_inline_execute"].asBool();
  }
  if (enable_inline_execute) {
    if (setenv("RUN_ALL_KERNELS_INLINE", "1", 1) != 0) {
      LOG(WARNING) << "Set RUN_ALL_KERNELS_INLINE env error: "
                   << json_config["enable_inline_execute"].asBool();
    }
  }
 
  if (!json_config["omp_num_threads"].isNull()) {
    if (setenv("OMP_NUM_THREADS",
               json_config["omp_num_threads"].asString().c_str(), 1) != 0) {
      LOG(WARNING) << "Set OMP_NUM_THREADS env error: "
                   << json_config["omp_num_threads"];
    }
  }

  if (!json_config["kmp_blocktime"].isNull()) {
    if (setenv("KMP_BLOCKTIME",
               json_config["kmp_blocktime"].asString().c_str(), 1) != 0) {
      LOG(WARNING) << "Set KMP_BLOCKTIME env error: "
                   << json_config["kmp_blocktime"];
    }
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

  if (!json_config["model_update_inter_threads"].isNull()) {
    (*config)->model_update_inter_threads =
      json_config["model_update_inter_threads"].asInt();
  } else {
    (*config)->model_update_inter_threads = 0;
  }

  if (!json_config["model_update_intra_threads"].isNull()) {
    (*config)->model_update_intra_threads =
      json_config["model_update_intra_threads"].asInt();
  } else {
    (*config)->model_update_intra_threads = 0;
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

  if (!json_config["warmup_file_name"].isNull()) {
    (*config)->warmup_file_name =
      json_config["warmup_file_name"].asString();
  } else {
    (*config)->warmup_file_name = "";
  }

  if (!json_config["serialize_protocol"].isNull()) {
    (*config)->serialize_protocol =
      json_config["serialize_protocol"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No serialize_protocol in ModelConfig.");
  }

  if ((*config)->serialize_protocol.empty()) {
    return Status(error::Code::INVALID_ARGUMENT,
        "[TensorFlow] serialize_protocol shouldn't be empty string.");
  }

  if (!json_config["checkpoint_dir"].isNull()) {
    (*config)->enable_incr_model_update = true;
    (*config)->checkpoint_dir =
      json_config["checkpoint_dir"].asString();
  } else {
    (*config)->enable_incr_model_update = false;
    LOG(WARNING) << "[TensorFlow] Disable increment model update, "
                 << "processor only load saved model.";
  }

  if (!json_config["savedmodel_dir"].isNull()) {
    (*config)->savedmodel_dir =
      json_config["savedmodel_dir"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No savedmodel_dir in ModelConfig.");
  }
  
  if (!json_config["feature_store_type"].isNull()) {
    (*config)->feature_store_type =
      json_config["feature_store_type"].asString();
  } else {
    return Status(error::Code::NOT_FOUND,
        "[TensorFlow] No feature_store_type in ModelConfig.");
  }

  if ((*config)->feature_store_type.empty()) {
    return Status(error::Code::INVALID_ARGUMENT,
        "[TensorFlow] feature_store_type shouldn't be empty string.");
  }

  // @feature_store_type: 
  // 'redis/cluster_redis' or 'local'
  if ((*config)->feature_store_type == "cluster_redis" ||
      (*config)->feature_store_type == "redis") {
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
    if (!json_config["checkpoint_dir"].isNull() &&
        (*config)->checkpoint_dir.find((*config)->model_store_type)
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

    if (!json_config["checkpoint_dir"].isNull()) {
      TF_RETURN_IF_ERROR(AddOSSAccessPrefix(
            (*config)->checkpoint_dir, *config));
    }
  }

  // timeout of distribute lock
  if (!json_config["lock_timeout"].isNull()) {
    (*config)->lock_timeout =
      json_config["lock_timeout"].asInt();
  } else {
    (*config)->lock_timeout = 15 * 60; // 900 seconds
  }

  (*config)->use_per_session_threads = false;
  if (!json_config["use_per_session_threads"].isNull()) {
    (*config)->use_per_session_threads =
        json_config["use_per_session_threads"].asBool();
  }

  (*config)->shard_embedding = false;
  bool shard_embedding = false;
  if (!json_config["shard_embedding"].isNull()) {
    shard_embedding = json_config["shard_embedding"].asBool();
  }

  if (shard_embedding) {
    if ((*config)->feature_store_type != "local") {
      return Status(error::Code::INVALID_ARGUMENT,
          "[TensorFlow] Sharded embedding must be load in local,"
          "this require feature_store_type must be 'local' mode.");
    }

    (*config)->shard_embedding = true;
    if (json_config["embedding_names"].isNull() ||
        json_config["shard_instance_count"].isNull() ||
        json_config["id_type"].isNull()) {
        return Status(error::Code::INVALID_ARGUMENT,
            "[TensorFlow] Shard embedding require args: embedding_names, "
            "shard_instance_count and id_type.");
    }

    std::string embedding_names = json_config["embedding_names"].asString();
    (*config)->shard_instance_count = json_config["shard_instance_count"].asInt();
    // "string" or "int64"
    (*config)->id_type = json_config["id_type"].asString();

    // "name1;name2;name3"
    auto idx = embedding_names.find(";");
    while (idx != std::string::npos) {
      (*config)->shard_embedding_names.push_back(embedding_names.substr(0, idx));
      embedding_names = embedding_names.substr(idx+1);
      idx = embedding_names.find(";");
    }
    (*config)->shard_embedding_names.push_back(embedding_names);
  }

  // enable trace timeline
  if (!json_config["timeline_start_step"].isNull() &&
      !json_config["timeline_interval_step"].isNull() &&
      !json_config["timeline_trace_count"].isNull() &&
      !json_config["timeline_path"].isNull()) {
    auto path = json_config["timeline_path"].asString();
    auto start_step = json_config["timeline_start_step"].asInt();
    auto interval_step = json_config["timeline_interval_step"].asInt();
    auto trace_count = json_config["timeline_trace_count"].asInt();
 
    // save timeline to local
    if (path[0] == '/') {
      Tracer::GetTracer()->SetParams(start_step, interval_step, trace_count, path);
    } else if (path.find("oss://") != std::string::npos) {
      // save timeline to oss
      if ((*config)->oss_endpoint == "" ||
          (*config)->oss_access_id == "" ||
          (*config)->oss_access_key == "") {
        return Status(error::Code::INVALID_ARGUMENT,
            "Timeline require oss_endpoint, oss_access_id, and oss_access_key.");
      }
      Tracer::GetTracer()->SetParams(start_step,
          interval_step, trace_count, (*config)->oss_endpoint,
          (*config)->oss_access_id, (*config)->oss_access_key, path);
    } else {
      return Status(error::Code::INVALID_ARGUMENT,
          "[TensorFlow] Only support save timeline to local or oss now.");
    }
  }

  if (!json_config["ev_storage_type"].isNull()) {
    auto st = json_config["ev_storage_type"].asInt();
    switch (st) {
      case embedding::StorageType::DEFAULT:
        break;
      case embedding::StorageType::DRAM:
        (*config)->storage_type = embedding::StorageType::DRAM;
        break;
      case embedding::StorageType::DRAM_SSDHASH:
        (*config)->storage_type = embedding::StorageType::DRAM_SSDHASH;
        (*config)->storage_path = json_config["ev_storage_path"].asString();
        for (int i = 0; i < json_config["ev_storage_size"].size(); i++)
          (*config)->storage_size.emplace_back(json_config["ev_storage_size"][i].asInt64());
        if (json_config["ev_storage_size"].size() < 4) {
          for (int i =  json_config["ev_storage_size"].size(); i < 4; i++)
            (*config)->storage_size.emplace_back(1024*1024*1024);
        }
        break;
      default:
        return Status(error::Code::INVALID_ARGUMENT,
          "[TensorFlow] Only support ev storage type {DRAM, DRAM_SSDHASH}.");
    }
  }

  return Status::OK();
}

} // processor
} // tensorflow

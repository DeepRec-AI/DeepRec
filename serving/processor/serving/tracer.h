#ifndef SERVING_PROCESSOR_SERVING_TRACER_H
#define SERVING_PROCESSOR_SERVING_TRACER_H

#include <fstream>
#include <iostream>
#include <string>
#include "aos_http_io.h"
#include "oss_api.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace processor {

enum class TimelineLocation {
  LOCAL,
  OSS
};

class Tracer {
 public:
  static Tracer* GetTracer() {
    static Tracer t;
    return &t;
  }

  ~Tracer() {
    if (tracing_ && location_type_ == TimelineLocation::OSS) {
      aos_pool_destroy(pool_);
      aos_http_io_deinitialize();
    }
  }

  Tracer() : tracing_(false), curr_step_(0) {}
  Tracer(int64_t start_step,
         int64_t interval_step, 
         int64_t tracing_count,
         const std::string& oss_endpoint,
         const std::string& oss_access_id,
         const std::string& oss_access_secret,
         const std::string& path)
    : tracing_(true),
      next_tracing_step_(start_step),
      interval_step_(interval_step),
      tracing_count_(tracing_count),
      curr_step_(0),
      oss_endpoint_(oss_endpoint),
      access_key_id_(oss_access_id),
      access_key_secret_(oss_access_secret) {
    location_type_ = TimelineLocation::OSS;
    limit_step_ = start_step +
        interval_step * tracing_count;
    ParseFilePath(path);
    PrintDebugString();

    InitOss();
  }

  void SetParams(const int64_t start_step,
                 const int64_t interval_step,
                 const int64_t tracing_count,
                 const std::string& oss_endpoint,
                 const std::string& oss_access_id,
                 const std::string& oss_access_secret, 
                 const std::string& path) {
    location_type_ = TimelineLocation::OSS;
    tracing_ = true;
    next_tracing_step_ = start_step;
    interval_step_ = interval_step;
    tracing_count_ = tracing_count;
    limit_step_ = start_step +
        interval_step * tracing_count;
    oss_endpoint_ = oss_endpoint;
    access_key_id_ = oss_access_id;
    access_key_secret_ = oss_access_secret;
    ParseFilePath(path);
    PrintDebugString();

    InitOss();
  }

  void SetParams(const int64_t start_step,
                 const int64_t interval_step,
                 const int64_t tracing_count,
                 const std::string& path) {
    location_type_ = TimelineLocation::LOCAL;
    tracing_ = true;
    next_tracing_step_ = start_step;
    interval_step_ = interval_step;
    tracing_count_ = tracing_count;
    limit_step_ = start_step +
        interval_step * tracing_count;
    ParseFilePath(path);
    PrintDebugString();
  }


  bool NeedTracing() {
    if (!tracing_) return false;

    if (curr_step_ < limit_step_) {
      int64_t s = curr_step_.fetch_add(1, std::memory_order_relaxed);
      if (s == next_tracing_step_) {
        mutex_lock lock(mu_);
        next_tracing_step_ += interval_step_;
        return true;
      }
    }

    return false;
  }

  void GenTimeline(tensorflow::RunMetadata& run_metadata) {
    static std::atomic<int> counter(0);
    int index = counter.fetch_add(1, std::memory_order_relaxed);
    std::string outfile;
    run_metadata.step_stats().SerializeToString(&outfile);
    string file_name = file_path_dir_ + "timeline-" +
        std::to_string(index);
 
    if (location_type_ == TimelineLocation::LOCAL) {
      std::ofstream ofs;
      ofs.open(file_name);
      ofs << outfile;
      ofs.close();
    } else if (location_type_ == TimelineLocation::OSS) {
      aos_string_t object;
      aos_str_set(&object, file_name.c_str());
      aos_table_t* headers = aos_table_make(pool_, 0);
      aos_list_t buffer;
      aos_list_init(&buffer);
      aos_buf_t* content = aos_buf_pack(pool_, outfile.c_str(), outfile.length());
      aos_list_add_tail(&content->node, &buffer);
      aos_table_t *resp_headers;
      aos_status_t* resp_status =
          oss_put_object_from_buffer(oss_client_options_, &aos_bucket_name_,
                                     &object, &buffer, headers,
                                     &resp_headers);
      if (!aos_status_is_ok(resp_status)) {
        LOG(ERROR) << "Push timeline file fail: " << file_name;
      }
    }
  }

 private:
  void ParseFilePath(const std::string& path) {
    // Local
    if (path[0] == '/') {
      file_path_dir_ = path;
      file_path_dir_ += "/";
    } else if (path.find("oss://") != std::string::npos) {
      bucket_name_ = path.substr(6);
      auto pos = bucket_name_.find("/");
      if (pos == std::string::npos) {
        LOG(FATAL) << "Valid oss path must be start with oss://bucket/xxx/";
      }
      file_path_dir_ = bucket_name_.substr(pos+1);
      if (file_path_dir_[file_path_dir_.size()-1] != '/') {
        file_path_dir_ += "/";
      }
      bucket_name_ = bucket_name_.substr(0, pos);

      aos_str_set(&aos_bucket_name_, bucket_name_.c_str());
      aos_str_set(&aos_file_path_dir_, file_path_dir_.c_str());
    } else {
      LOG(FATAL) << "Valid path must be start with oss or local path.";
    }
  }

  void InitOptions(oss_request_options_t *options) {
    options->config = oss_config_create(options->pool);
    aos_str_set(&options->config->endpoint, oss_endpoint_.c_str());
    aos_str_set(&options->config->access_key_id, access_key_id_.c_str());
    aos_str_set(&options->config->access_key_secret, access_key_secret_.c_str());
    options->config->is_cname = 0;
    options->ctl = aos_http_controller_create(options->pool, 0);
  }

  void InitOss() {
    if (aos_http_io_initialize(NULL, 0) != AOSE_OK) {
      LOG(FATAL) << "Init oss env failed.";
    }

    aos_pool_create(&pool_, NULL);
    oss_client_options_ = oss_request_options_create(pool_);
    InitOptions(oss_client_options_);
  }

  void PrintDebugString() {
    LOG(INFO) << "tracing_: " << tracing_
              << ", next_tracing_step_: " << next_tracing_step_
              << ", interval_step_: " << interval_step_
              << ", tracing_count_: " << tracing_count_
              << ", limit_step_: " << limit_step_
              << ", bucket_name_: " << bucket_name_
              << ", file_path_dir_: " << file_path_dir_;
  }

 private:
  bool tracing_ = false;
  int64_t next_tracing_step_ = 0;
  int64_t interval_step_ = 1;
  int64_t tracing_count_ = 0;
  int64_t limit_step_ = 0;
  std::atomic<int64_t> curr_step_;
  TimelineLocation location_type_ = TimelineLocation::LOCAL;

  // oss info
  std::string oss_endpoint_ = "";
  std::string access_key_id_ = "";
  std::string access_key_secret_ = "";
  std::string bucket_name_ = "";
  std::string file_path_dir_ = "";

  aos_string_t aos_bucket_name_;
  aos_string_t aos_file_path_dir_;
  aos_pool_t* pool_;
  oss_request_options_t* oss_client_options_;

  mutex mu_;
};

} // namespace processor
} // namespace tensorflow

#endif // SERVING_PROCESSOR_SERVING_TRACER_H

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_GPU_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_GPU_UTILS_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <unordered_map>

#include "absl/types/span.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#if CUDNN_VERSION >= 8100
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#endif // CUDNN_VERSION >= 8100
#endif // GOOGLE_CUDA

namespace stream_executor {
namespace cuda {
class RedzoneAllocator;
}
}  // namespace stream_executor

namespace tensorflow {

class NodeDef;
class AutotuneResult;
class AutotuneExecutionPlanResult;

// Return whether the redzone check is disabled.
//
// Controlled by the TF_DISABLE_RZ_CHECK environment variable.
bool RedzoneCheckDisabled();

// Return an allocated buffer with redzones the size of `buffer`. Does
// *not* copy the contents of the `buffer` into the newly allocated buffer:
// assumes that buffer is a pure out-parameter.
//
// Returns `buffer` if RedzoneCheckDisabled() is true.
//
// On error, return `buffer`, and log an error message (once).
se::DeviceMemoryBase WrapRedzoneBestEffort(
    se::cuda::RedzoneAllocator* rz_allocator, se::DeviceMemoryBase buffer);

// Check the passed allocator for redzone violations.
// If violations have occurred, mark the corresponding autotune result
// as a failure.
template<typename T>
void CheckRedzones(const se::cuda::RedzoneAllocator& rz_allocator,
                   T* autotune_result);

template <typename T>
inline se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory, uint64 size) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory), size * sizeof(T));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

// A helper class that looks up the best autotuned config from parameters.
// Due to the noisy nature of autotune, especially with multiple devices, it
// only accepts a config if its margin exceeds a threshold.
// For the same shape configs, if a new best config matches the previous best,
// they get promoted; otherwise, the winner gets demoted. This process stops
// when the winner's score exceeds the threshold.
// In a bad case when two configs are very close to each other and flips
// back and forth randomly, the expected number of experiments before autotune
// settles is O(threshold ^ 2). So we recommend that number of warmup runs
// for any benchmarks.
template <typename Parameters, typename Config>
class AutoTuneMap {
 public:
  bool Find(const Parameters& params, Config* config) const {
    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    if (iter == params_config_map_.end() ||
        (iter->second.score < min_score_threshold_ &&
         iter->second.count <= max_autotune_count_)) {
      return false;
    }
    *config = iter->second.config;
    return true;
  }
  void Insert(const Parameters& params, const Config& config) {
    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    int new_score = 0;
    if (iter == params_config_map_.end()) {
      // Create a new entry if params is new.
      VLOG(1) << GetActionSummary("creates", params, config);
      params_config_map_.insert(
          std::make_pair(params, ValueType{config, 1, 1}));
      new_score = 1;
    } else if (iter->second.score < min_score_threshold_ &&
               iter->second.count <= max_autotune_count_) {
      DCHECK_GT(iter->second.score, 0);
      if (iter->second.config != config) {
        // If it is different from the current winner, demotes the winner.
        VLOG(1) << GetActionSummary("demotes", params, config);
        new_score = --iter->second.score;
        ++iter->second.count;
        if (new_score <= 0) {
          VLOG(1) << GetActionSummary("erases", params, config);
          params_config_map_.erase(iter);
        }
      } else {
        // If it is the same as the current winner, promotes the winner.
        VLOG(1) << GetActionSummary("promotes", params, config);
        new_score = ++iter->second.score;
        ++iter->second.count;
      }
    }
    if (new_score >= min_score_threshold_) {
      VLOG(1) << GetActionSummary("accepts", params, config);
    } else if (autotune_global_count_ >= max_autotune_global_count_) {
      // The autotuning exceeds the max iteration threshold and we accept the
      // the winner if it exists in the map, otherwise we accept the current
      // winner.
      auto winner = params_config_map_.find(params);
      if (winner == params_config_map_.end()) {
        VLOG(1) << GetActionSummary("creates", params, config);
        for (int i = 0; i < min_score_threshold_; ++i) {
          VLOG(1) << GetActionSummary("promotes", params, config);
        }
        params_config_map_.insert(
            std::make_pair(params, ValueType{config, min_score_threshold_, 1}));
      } else {
        int promotes_times = min_score_threshold_ - winner->second.score;
        for (int i = 0; i < promotes_times; ++i) {
          VLOG(1) << GetActionSummary("promotes", params, config);
        }
        winner->second.score = min_score_threshold_;
      }
      VLOG(1) << GetActionSummary("accepts", params, config);
    }
    autotune_global_count_++;
  }

 private:
  AutoTuneMap(const string& name) : name_(name) {
    min_score_threshold_ = 1;
    int min_warmup_iterations = 10;
    const char* threshold_str = getenv("TF_AUTOTUNE_THRESHOLD");
    if (threshold_str != nullptr) {
      strings::safe_strto32(threshold_str, &min_score_threshold_);
    }
    const char* min_warmup_iteration_str =
        getenv("TF_AUTOTUNE_MIN_WARMUP_ITERATIONS");
    if (min_warmup_iteration_str != nullptr) {
      strings::safe_strto32(min_warmup_iteration_str, &min_warmup_iterations);
    }
    min_score_threshold_ = std::max(min_score_threshold_, 1);
    max_autotune_count_ = std::max(
        5 * min_score_threshold_ * min_score_threshold_, min_warmup_iterations);
    max_autotune_global_count_ = 2 * max_autotune_count_;
    autotune_global_count_ = 0;
  }

  template <class Group, class Params, class Cfg>
  friend class AutoTuneSingleton;

  struct Hasher {
    std::size_t operator()(const Parameters& parameter) const {
      return parameter.hash();
    }
  };

  string GetActionSummary(StringPiece action, const Parameters& params,
                          const Config& config) {
    return strings::Printf("autotune_map %s %s: %s -> (%s)", name_.c_str(),
                           string(action).c_str(), params.ToString().c_str(),
                           config.ToString().c_str());
  }

  mutable mutex mu_;
  struct ValueType {
    Config config;
    int32 score;
    int32 count;
  };
  std::unordered_map<Parameters, ValueType, Hasher> params_config_map_
      GUARDED_BY(mu_);
  string name_;
  int32 min_score_threshold_;
  int32 max_autotune_count_;
  int32 max_autotune_global_count_;
  int32 autotune_global_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(AutoTuneMap);
};

#if CUDNN_VERSION >= 8100
using se::dnn::ExecutionPlanDesc;
using se::dnn::ExecutionPlanConfig;
template <typename Parameters>
class AutoTuneExecutionPlanMap {
 public:
  bool Find(const Parameters& params, ExecutionPlanConfig* plan_config) {
    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    if (iter == params_config_map_.end() ||
        (iter->second.score < min_score_threshold_ &&
         iter->second.count <= max_autotune_count_)) {
      return false;
    }
    auto& plan = iter->second.plan;
    plan_config->set_plan(ExecutionPlanDesc(plan.getTag(), plan.get_raw_desc()));
    plan_config->set_scratch_size(plan.getWorkspaceSize());
    if (iter->second.plan_no_scratch.has_value()) {
      auto& plan_no_scratch = iter->second.plan_no_scratch;
      plan_config->set_plan_no_scratch(
          ExecutionPlanDesc(plan_no_scratch->getTag(),
                            plan_no_scratch->get_raw_desc()));
    }
    return true;
  }
  void Insert(const Parameters& params,
              std::vector<cudnn_frontend::ExecutionPlan>& plans) {
    if (plans.size() != 1 and plans.size() != 2) {
      return;
    }
    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    int new_score = 0;
    if (iter == params_config_map_.end()) {
      // Create a new entry if params is new.
      VLOG(1) << GetActionSummary("creates", params, plans);
      new_score = 1;
      UpdateMap(params, new_score, 1, plans);
    } else if (iter->second.score < min_score_threshold_ &&
               iter->second.count <= max_autotune_count_) {
      DCHECK_GT(iter->second.score, 0);

      bool is_diff;
      ExecutionPlanDesc old_plan(iter->second.plan.getTag(),
                                 iter->second.plan.get_raw_desc());
      ExecutionPlanDesc new_plan(plans[0].getTag(), plans[0].get_raw_desc());
      if (plans.size() == 1) {
        ExecutionPlanConfig old_plan_config(
            old_plan, iter->second.plan.getWorkspaceSize());
        ExecutionPlanConfig new_plan_config(
            new_plan, plans[0].getWorkspaceSize());
        is_diff = new_plan_config != old_plan_config;
      } else if (iter->second.plan_no_scratch.has_value()) {
        ExecutionPlanDesc old_plan_no_scratch(
            iter->second.plan_no_scratch->getTag(),
            iter->second.plan_no_scratch->get_raw_desc());
        ExecutionPlanDesc new_plan_no_scratch(
            plans[1].getTag(), plans[1].get_raw_desc());
        ExecutionPlanConfig old_plan_config(old_plan,
            iter->second.plan.getWorkspaceSize(), old_plan_no_scratch);
        ExecutionPlanConfig new_plan_config(
            new_plan, plans[1].getWorkspaceSize(), new_plan_no_scratch);
        is_diff = new_plan_config != old_plan_config;
      } else {
        is_diff = false;
      }

      if (is_diff) {
        // If it is different from the current winner, demotes the winner.
        VLOG(1) << GetActionSummary("demotes", params, iter->second);
        new_score = --iter->second.score;
        auto new_count = ++iter->second.count;
        if (new_score <= 0) {
          VLOG(1) << GetActionSummary("erases", params, iter->second);
          params_config_map_.erase(iter);
        }
      } else {
        // If it is the same as the current winner, promotes the winner.
        VLOG(1) << GetActionSummary("promotes", params, iter->second);
        new_score = ++iter->second.score;
        auto new_count = ++iter->second.count;
      }
    }
    if (new_score >= min_score_threshold_) {
      VLOG(1) << GetActionSummary("accepts", params, iter->second);
    } else if (autotune_global_count_ >= max_autotune_global_count_) {
      // The autotuning exceeds the max iteration threshold and we accept the
      // the winner if it exists in the map, otherwise we accept the current
      // winner.
      auto winner = params_config_map_.find(params);
      if (winner == params_config_map_.end()) {
        VLOG(1) << GetActionSummary("creates", params, plans);
        for (int i = 0; i < min_score_threshold_; ++i) {
          VLOG(1) << GetActionSummary("promotes", params, plans);
        }
        // Log the plans before the UpdateMap(), which will move the objects out
        // of the vector.
        VLOG(1) << GetActionSummary("accepts", params, plans);
        UpdateMap(params, min_score_threshold_, 1, plans);
      } else {
        int promotes_times = min_score_threshold_ - winner->second.score;
        for (int i = 0; i < promotes_times; ++i) {
          VLOG(1) << GetActionSummary("promotes", params, winner->second);
        }
        winner->second.score = min_score_threshold_;
        VLOG(1) << GetActionSummary("accepts", params, winner->second);
      }
    }
    autotune_global_count_++;
  }

 private:
  void UpdateMap(const Parameters& params, int score, int count,
                 std::vector<cudnn_frontend::ExecutionPlan>& plans) {
    if (plans.size() == 1) {
      params_config_map_.insert(std::make_pair(params,
          ValueType{std::move(plans[0]), {}, score, count}));
    } else {
      params_config_map_.insert(std::make_pair(params,
          ValueType{std::move(plans[0]), std::move(plans[1]), score, count}));
    }
  }
  AutoTuneExecutionPlanMap(const string& name) : name_(name) {
    min_score_threshold_ = 1;
    int min_warmup_iterations = 10;
    const char* threshold_str = getenv("TF_AUTOTUNE_THRESHOLD");
    if (threshold_str != nullptr) {
      strings::safe_strto32(threshold_str, &min_score_threshold_);
    }
    const char* min_warmup_iteration_str =
        getenv("TF_AUTOTUNE_MIN_WARMUP_ITERATIONS");
    if (min_warmup_iteration_str != nullptr) {
      strings::safe_strto32(min_warmup_iteration_str, &min_warmup_iterations);
    }
    min_score_threshold_ = std::max(min_score_threshold_, 1);
    max_autotune_count_ = std::max(
        5 * min_score_threshold_ * min_score_threshold_, min_warmup_iterations);
    max_autotune_global_count_ = 2 * max_autotune_count_;
    autotune_global_count_ = 0;
  }

  template <class Group, class Params>
  friend class AutoTuneExecutionPlanSingleton;

  struct Hasher {
    std::size_t operator()(const Parameters& parameter) const {
      return parameter.hash();
    }
  };

  string GetActionSummary(StringPiece action, const Parameters& params,
                          std::vector<cudnn_frontend::ExecutionPlan>& plans) {
    std::string plan_str = plans[0].getTag();
    std::string plan_no_scratch_str = "none";
    if (plans.size() > 1) {
      plan_no_scratch_str = plans[1].getTag();
    }
    return strings::Printf("autotune_map %s %s: %s -> (%s, %s)", name_.c_str(),
                           string(action).c_str(), params.ToString().c_str(),
                           plan_str.c_str(), plan_no_scratch_str.c_str());
  }

  mutable mutex mu_;
  struct ValueType {
    cudnn_frontend::ExecutionPlan plan;
    absl::optional<cudnn_frontend::ExecutionPlan> plan_no_scratch;
    int32 score;
    int32 count;
  };
  string GetActionSummary(StringPiece action, const Parameters& params,
                          ValueType& value) {
    std::string plan_str = value.plan.getTag();
    std::string plan_no_scratch_str = "none";
    if (value.plan_no_scratch.has_value()) {
      plan_no_scratch_str = value.plan_no_scratch->getTag();
    }
    return strings::Printf("autotune_map %s %s: %s -> (%s, %s)", name_.c_str(),
                           string(action).c_str(), params.ToString().c_str(),
                           plan_str.c_str(), plan_no_scratch_str.c_str());
  }
  std::unordered_map<Parameters, ValueType, Hasher> params_config_map_
      TF_GUARDED_BY(mu_);
  string name_;
  int32 min_score_threshold_;
  int32 max_autotune_count_;
  int32 max_autotune_global_count_;
  int32 autotune_global_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(AutoTuneExecutionPlanMap);
};
#endif // CUDNN_VERSION >= 8100

// A Singleton helper that manages the global autotune results by groups.
// The caller specified arbitrary Group type that can distinguish between
// different autotune results, even if their Parameters and Configs are the
// same.
template <class Group, typename Parameters, typename Config>
class AutoTuneSingleton {
 public:
  typedef AutoTuneMap<Parameters, Config> AutoTuneType;
  static AutoTuneType* GetInstance() {
    static AutoTuneType* instance = new AutoTuneType(Group::name());
    return instance;
  }
};

#if CUDNN_VERSION >= 8100
template <class Group, typename Parameters>
class AutoTuneExecutionPlanSingleton {
 public:
  typedef AutoTuneExecutionPlanMap<Parameters> AutoTuneType;
  static AutoTuneType* GetInstance() {
    static AutoTuneType* instance = new AutoTuneType(Group::name());
    return instance;
  }
};
#endif // CUDNN_VERSION >= 8100

// Logs convolution results to customized back-storage.
void LogConvAutotuneResults(se::dnn::ConvolutionKind kind,
                            se::dnn::DataType element_type,
                            se::DeviceMemoryBase input_buffer,
                            se::DeviceMemoryBase filter_buffer,
                            se::DeviceMemoryBase output_buffer,
                            const se::dnn::BatchDescriptor& input_desc,
                            const se::dnn::FilterDescriptor& filter_desc,
                            const se::dnn::BatchDescriptor& output_desc,
                            const se::dnn::ConvolutionDescriptor& conv_desc,
                            se::StreamExecutor* stream_exec,
                            absl::Span<const AutotuneResult> results);

void LogConvAutotuneResults(se::dnn::ConvolutionKind kind,
    se::dnn::DataType element_type, se::DeviceMemoryBase input_buffer,
    se::DeviceMemoryBase filter_buffer, se::DeviceMemoryBase output_buffer,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    se::StreamExecutor* stream_exec,
    absl::Span<const AutotuneExecutionPlanResult> results);

// Logs fused convolution results to customized back-storage.
void LogFusedConvForwardAutotuneResults(
    se::dnn::DataType element_type, se::DeviceMemoryBase input_buffer,
    se::DeviceMemoryBase filter_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase bias_buffer, se::DeviceMemoryBase side_input_buffer,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc, double conv_scale,
    double side_value_scale, se::dnn::ActivationMode activation_mode,
    se::StreamExecutor* stream_exec, absl::Span<const AutotuneResult> results);

void LogFusedConvForwardAutotuneResults(
    se::dnn::DataType element_type, se::DeviceMemoryBase input_buffer,
    se::DeviceMemoryBase filter_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase bias_buffer, se::DeviceMemoryBase side_input_buffer,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc, double conv_scale,
    double side_value_scale, se::dnn::ActivationMode activation_mode,
    se::StreamExecutor* stream_exec,
    absl::Span<const AutotuneExecutionPlanResult> results);

// Returns the best algorithms for the config, one is the fastest, the other is
// other is fastest with 0 scracth space. Unsuccessful autotuning results are
// allowed and ignored.
Status BestCudnnConvAlgorithm(absl::Span<const AutotuneResult> results,
                              se::dnn::AlgorithmConfig* algo);

Status BestCudnnConvExecutionPlan(
    absl::Span<const AutotuneExecutionPlanResult> results,
    int* idx_plan, int* idx_plan_no_scratch);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_UTILS_H_

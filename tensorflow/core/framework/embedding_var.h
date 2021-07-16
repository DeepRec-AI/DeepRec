/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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


#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VAR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VAR_H_


#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/framework/hashmap.h"

namespace tensorflow {

template <class K, class V>
class EmbeddingVar : public ResourceBase {
 public:
  EmbeddingVar(const string& name, HashMap<K, V>* hash_map,
      int64 steps_to_live = 0)
  : name_(name), hash_map_(hash_map), steps_to_live_(steps_to_live) {}

  Status Init(const Tensor& default_tensor) {
    return hash_map_->Init(default_tensor);
  }

  HashMap<K, V>* hashmap() { return hash_map_; }

  string DebugString() const {
    return name_;
  }
  void SetInitialized() {
    is_initialized_ = true;
  }
  bool IsInitialized() const {
    return is_initialized_;
  }

  Status Shrink(int64 gs) {
    if (steps_to_live_ > 0) {
      return hash_map_->Shrink(steps_to_live_, gs);
    }
    return Status::OK();
  }

  Status ExportValues(OpKernelContext* ctx) {
    std::vector<K> key_list;
    std::vector<V* > valueptr_list;
    std::vector<int64> version_list;
    int64 total_size = hash_map_->GetSnapshot(&key_list, &valueptr_list, &version_list);
    LOG(INFO) << "EV Export size:" << total_size;

    Tensor* key = nullptr;
    Tensor* val = nullptr;
    Tensor* version = nullptr;
    TF_RETURN_IF_ERROR(ctx->allocate_output(0, TensorShape({total_size}), &key));
    TF_RETURN_IF_ERROR(ctx->allocate_output(1, TensorShape({total_size, hash_map_->ValueLen()}), &val));
    TF_RETURN_IF_ERROR(ctx->allocate_output(2, TensorShape({total_size}), &version));
    auto key_flat = key->flat<K>();
    auto val_matrix = val->matrix<V>();
    auto version_flat = version->flat<int64>();


    int64 ii = 0;
    for (size_t i = 0; i < key_list.size(); i++) {
      K key = key_list[i];
      V* val = valueptr_list[i];

      key_flat(ii) = key;

      Eigen::array<Eigen::DenseIndex, 1> dims({hash_map_->ValueLen()});
      typename TTypes<V>::Flat value_flat = typename TTypes<V>::Flat(val, dims);

      for (int64 j = 0; j < hash_map_->ValueLen(); ++j) {
          val_matrix(ii, j) = value_flat(j);
      }
      int64 dump_version = *(reinterpret_cast<int64*>(val + hash_map_->ValueLen()));
      version_flat(ii) = dump_version;
      ii++;
    }
    return Status::OK();
  }

  mutex* mu() { return &mu_; }

 private:
  std::string name_;
  mutex mu_;
  HashMap<K, V>* hash_map_;
  int64 steps_to_live_;
  bool is_initialized_ = false;

  ~EmbeddingVar() override {
    delete hash_map_;
  }
  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVar);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VAR_H_



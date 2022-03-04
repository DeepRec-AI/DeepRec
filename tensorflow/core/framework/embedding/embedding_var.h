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


#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/framework/embedding/embedding_filter.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/multilevel_embedding.h"
#include "tensorflow/core/framework/typed_allocator.h"

namespace tensorflow {

struct RestoreBuffer {
  char* key_buffer;
  char* value_buffer;
  char* version_buffer;
  char* freq_buffer;

  ~RestoreBuffer() {
    delete key_buffer;
    delete value_buffer;
    delete version_buffer;
    delete freq_buffer;
  }
};

template <class K, class V>
class EmbeddingVar : public ResourceBase {
 public:
  EmbeddingVar(const string& name,
               embedding::StorageManager<K, V>* storage_manager,
               EmbeddingConfig emb_cfg = EmbeddingConfig()):
      name_(name),
      storage_manager_(storage_manager),
      default_value_(nullptr),
      value_len_(0),
      alloc_(nullptr),
      emb_config_(emb_cfg) {}

  Status Init(const Tensor& default_tensor, int64 default_value_dim) {
    filter_ = FilterFactory::CreateFilter<K, V, EmbeddingVar<K, V>>(emb_config_, this);

    // for default value allocation
    alloc_ = cpu_allocator();

    if (storage_manager_ == nullptr) {
      return errors::InvalidArgument("Invalid ht_type to construct EmbeddingVar");
    } else {
      emb_config_.default_value_dim = default_value_dim;
      value_len_ = default_tensor.NumElements()/emb_config_.default_value_dim;
      default_value_ = TypedAllocator::Allocate<V>(alloc_, default_tensor.NumElements(), AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      memcpy(default_value_, &default_tensor_flat(0), default_tensor.TotalBytes());
      if (LayoutType::NORMAL_FIX == emb_config_.get_layout_type()) {
        storage_manager_->SetDim(emb_config_.emb_index, value_len_, emb_config_.slot_num + 1);
      }
      return Status::OK();
    }
  }

  void SetInitialized() {
    is_initialized_ = true;
  }

  bool IsInitialized() const {
    return is_initialized_;
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr, bool* is_filter,
      int64 update_version = -1) {
    return filter_->LookupOrCreateKey(key, value_ptr, is_filter, update_version);
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr, int64 update_version = -1) {
    Status s = storage_manager_->GetOrCreate(key, value_ptr, emb_config_.total_num(storage_manager_->GetTotalDims()));
    TF_CHECK_OK(s);
    if (emb_config_.is_primary() && emb_config_.steps_to_live != 0 && update_version != -1) {
      (*value_ptr)->SetStep(update_version);
    }
    return s;
  }

  void BatchCommit(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    TF_CHECK_OK(storage_manager_->BatchCommit(keys, value_ptrs));
  }

  int64 GetVersion(K key) {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(LookupOrCreateKey(key, &value_ptr));
    return value_ptr->GetStep();
  }

  int64 GetFreq(K key) {
    return filter_->GetFreq(key);
  }

  void LookupOrCreate(K key, V* val, V* default_v)  {
    const V* default_value_ptr = (default_v == nullptr) ? default_value_ : default_v;
    filter_->LookupOrCreate(key, val, default_value_ptr);
  }

  void LookupOrCreateWithFreq(K key, V* val, V* default_v)  {
    const V* default_value_ptr = (default_v == nullptr) ? default_value_ : default_v;
    filter_->LookupOrCreateWithFreq(key, val, default_value_ptr);
  }

  void LookupOrCreate(K key, V* val, V* default_v, int64 count)  {
    const V* default_value_ptr = (default_v == nullptr) ? default_value_ : default_v;
    filter_->LookupOrCreate(key, val, default_value_ptr, count);
  }

  V* LookupOrCreateEmb(ValuePtr<V>* value_ptr, const V* default_v) {
    return value_ptr->GetOrAllocate(alloc_, value_len_, default_v,
        emb_config_.emb_index, storage_manager_->GetOffset(emb_config_.emb_index));
  }

  V* LookupPrimaryEmb(ValuePtr<V>* value_ptr) {
    V* primary_val = value_ptr->GetValue(emb_config_.primary_emb_index, storage_manager_->GetOffset(emb_config_.primary_emb_index));
    return primary_val;
  }

  typename TTypes<V>::Flat flat(ValuePtr<V>* value_ptr) {
    V* val = LookupOrCreateEmb(value_ptr, default_value_);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }

  void Commit(const K id, ValuePtr<V>* value_ptr) {
    TF_CHECK_OK(storage_manager_->Commit(id, value_ptr));
  }

  int64 ValueLen() const {
    return value_len_;
  }

  int64 Size() const {
    return storage_manager_->Size();
  }

  int64 MinFreq() {
    return emb_config_.filter_freq;
  }

  int64 StepsToLive() const {
    return emb_config_.steps_to_live;
  }

  float GetL2WeightThreshold() {
    return emb_config_.l2_weight_threshold;
  }

  bool IsMultiLevel() {
    return emb_config_.is_multi_level;
  }

  std::string DebugString() const {
    return emb_config_.DebugString();
  }

  EmbeddingFilter<K, V, EmbeddingVar<K, V>>* GetFilter() const {
    return filter_;
  }

  Status Import(RestoreBuffer& restore_buff,
                int64 key_num,
                int bucket_num,
                int64 partition_id,
                int64 partition_num) {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    int64* version_buff = (int64*)restore_buff.version_buffer;
    int64* freq_buff = (int64*)restore_buff.freq_buffer;
    for (auto i = 0; i < key_num; ++i) {
      // this can describe by graph(Mod + DynamicPartition), but memory waste and slow
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        LOG(INFO) << "skip EV key:" << *(key_buff + i);
        continue;
      }
      ValuePtr<V>* value_ptr = nullptr;
      TF_CHECK_OK(LookupOrCreateKey(key_buff[i], &value_ptr));
      if (emb_config_.is_primary()) {
        if (emb_config_.filter_freq != 0) {
          if (freq_buff[i] <= emb_config_.filter_freq) {
            value_ptr->SetFreq(emb_config_.filter_freq);
          } else {
            value_ptr->SetFreq(freq_buff[i]);
          }
        }
        if (emb_config_.steps_to_live != 0) {
          value_ptr->SetStep(version_buff[i]);
        }
      }
      V* v = LookupOrCreateEmb(value_ptr, value_buff + i * value_len_);
      TF_CHECK_OK(storage_manager_->Commit(key_buff[i], value_ptr));
    }
    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list,
                    std::vector<int64>* version_list, std::vector<int64>* freq_list) {
    return storage_manager_->GetSnapshot(key_list, value_list, version_list,
                                         freq_list, emb_config_, filter_);
  }

  Status Destroy() {
    return storage_manager_->Destroy();
  }

  mutex* mu() {
    return &mu_;
  }

  embedding::StorageManager<K, V>* storage_manager() {
    return storage_manager_;
  }

  Status Shrink() {
    return storage_manager_->Shrink(emb_config_, value_len_);
  }

  Status Shrink(int64 gs) {
    if (emb_config_.steps_to_live > 0) {
      return storage_manager_->Shrink(gs, emb_config_.steps_to_live);
    }
  }

  V* GetDefaultValuePtr() {
    return default_value_;
  }

  int64 GetDefaultValueDim() {
    return emb_config_.default_value_dim;
  }

  void SetSlotNum(int64 slot_num) {
    emb_config_.slot_num = slot_num;
  }

  embedding::BatchCache<K>* Cache() {
    return storage_manager_->Cache();
  }

 private:
  std::string name_;
  bool is_initialized_ = false;

  mutex mu_;

  V* default_value_;
  int64 value_len_;
  Allocator* alloc_;
  embedding::StorageManager<K, V>* storage_manager_;
  EmbeddingConfig emb_config_;
  EmbeddingFilter<K, V, EmbeddingVar<K, V>>* filter_;

  ~EmbeddingVar() override {
    if (emb_config_.is_primary()) {
      Destroy();
      delete storage_manager_;
    }
    TypedAllocator::Deallocate(alloc_, default_value_, value_len_);
  }
  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVar);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_



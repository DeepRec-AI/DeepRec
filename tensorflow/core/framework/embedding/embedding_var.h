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
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"

namespace tensorflow {

struct RestoreBuffer {
  char* key_buffer;
  char* value_buffer;
  char* version_buffer;

  ~RestoreBuffer() {
    delete key_buffer;
    delete value_buffer;
    delete version_buffer;
  }
};

struct EmbeddingConfig {
  int64 emb_index;
  int64 primary_emb_index;
  int64 block_num;
  int64 slot_num;
  std::string name;
  int64 steps_to_live;
  int64 min_freq;

  EmbeddingConfig(int64 emb_index = 0, int64 primary_emb_index = 0,
                  int64 block_num = 1, int slot_num = 1,
                  const std::string& name = "", int64 steps_to_live = 0,
                  int64 min_freq = 0):
      emb_index(emb_index), primary_emb_index(primary_emb_index),
      block_num(block_num), slot_num(slot_num),
      name(name), steps_to_live(steps_to_live),
      min_freq(min_freq) {}

  bool is_primary() const {
    return emb_index == primary_emb_index;
  }

  int64 total_num() {
    return block_num * (slot_num + 1);
  }

  int64 get_min_freq() {
    return min_freq;
  }

  std::string DebugString() const {
    return strings::StrCat("opname: ", name,
                           " emb_index: ", emb_index,
                           " primary_emb_index: ", primary_emb_index,
                           " block_num: ", block_num,
                           " slot_num: ", slot_num);
  }
};

template <class K, class V>
class EmbeddingVar : public ResourceBase {
 public:
  EmbeddingVar(const string& name,
               KVInterface<K, V>* kv,
               Allocator* alloc = cpu_allocator(),
               EmbeddingConfig emb_cfg = EmbeddingConfig()):
      name_(name),
      kv_(kv),
      default_value_(nullptr),
      value_len_(0),
      alloc_(alloc),
      emb_config_(emb_cfg) {}

  Status Init(const Tensor& default_tensor) {
    if (default_tensor.dims() != 1) {
      return errors::InvalidArgument("EV's default_tensor shape must be 1-D");
    } else if (DataTypeToEnum<V>::v() != default_tensor.dtype()) {
       return errors::InvalidArgument("EV's default_tensor DTYPE must be same as EmbeddingVar Value Type");
    } else if (kv_ == nullptr) {
       return errors::InvalidArgument("Invalid ht_type to construct EmbeddingVar");
    } else {
      value_len_ = default_tensor.NumElements();
      default_value_ = TypedAllocator::Allocate<V>(alloc_, value_len_, AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      memcpy(default_value_, &default_tensor_flat(0), default_tensor.TotalBytes());
      return Status::OK();
    }
  }

  void SetInitialized() {
    is_initialized_ = true;
  }
  bool IsInitialized() const {
    return is_initialized_;
  }

  Status LookupOrCreateKeyInternal(K key, ValuePtr<V>** value_ptr, size_t size) {
    Status s = kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      // Found
      return s;
    } else {
      // Not found
      *value_ptr = new ValuePtr<V>(size);
      s = kv_->Insert(key, *value_ptr);
      if (s.ok()) {
        // Insert Success
        return s;
      } else {
        // Insert Failed, key already exist
        delete *value_ptr;
        s = kv_->Lookup(key, value_ptr);
        return s;
      }
    }
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr, int64 update_version = -1) {
    Status s = LookupOrCreateKeyInternal(key, value_ptr, emb_config_.total_num());
    TF_CHECK_OK(s);
    if (emb_config_.is_primary() && emb_config_.steps_to_live != 0 && update_version != -1) {
      (*value_ptr)->SetStep(update_version);
    }
    return s;
  }

  int64 GetVersion(K key) {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(LookupOrCreateKey(key, &value_ptr));
    return value_ptr->GetStep();
  }

  void LookupOrCreate(K key, V* val, V* default_v)  {
    const V* default_value_ptr = (default_v == nullptr) ? default_value_ : default_v;
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(LookupOrCreateKey(key, &value_ptr));
    value_ptr->AddFreq();
    if (value_ptr->GetFreq() >= emb_config_.min_freq) {
      V* mem_val = LookupOrCreateEmb(value_ptr, emb_config_, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * value_len_);
    } else {
      memcpy(val, default_value_ptr, sizeof(V) * value_len_);
    }
  }

  V* LookupOrCreateEmb(ValuePtr<V>* value_ptr, const EmbeddingConfig& embcfg, const V* default_v) {
    return value_ptr->GetOrAllocate(alloc_, value_len_, default_v, embcfg.emb_index);
  }

  typename TTypes<V>::Flat flat(ValuePtr<V>* value_ptr) {
    V* val = LookupOrCreateEmb(value_ptr, emb_config_, default_value_);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }

  int64 ValueLen() const {
    return value_len_;
  }

  int64 Size() const {
    return kv_->Size();
  }

  int64 MinFreq() {
    return emb_config_.min_freq;
  }

  std::string DebugString() const {
    return kv_->DebugString();
  }

  Status Import(RestoreBuffer& restore_buff,
                int64 key_num,
                int bucket_num,
                int64 partition_id,
                int64 partition_num) {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    int64* version_buff = (int64*)restore_buff.version_buffer;
    for (auto i = 0; i < key_num; ++i) {
      // this can describe by graph(Mod + DynamicPartition), but memory waste and slow
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        LOG(INFO) << "skip EV key:" << *(key_buff + i);
        continue;
      }
      ValuePtr<V>* value_ptr = nullptr;
      TF_CHECK_OK(LookupOrCreateKey(key_buff[i], &value_ptr, version_buff[i]));
      if (value_ptr->GetFreq() >= emb_config_.min_freq) {
        LookupOrCreateEmb(value_ptr, emb_config_, value_buff + i * value_len_);
      }
    }
    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list, std::vector<int64>* version_list) {
    std::vector<ValuePtr<V>* > value_ptr_list;
    std::vector<K> key_list_tmp;
    kv_->GetSnapshot(&key_list_tmp, &value_ptr_list);
    for (int64 i = 0; i < key_list_tmp.size(); ++i) {
      V* val = value_ptr_list[i]->GetValue(emb_config_.emb_index);
      V* primary_val = value_ptr_list[i]->GetValue(emb_config_.primary_emb_index);
      if (val != nullptr && primary_val != nullptr) {
        value_list->push_back(val);
        key_list->push_back(key_list_tmp[i]);
        if (emb_config_.steps_to_live != 0) {
          int64 dump_version = value_ptr_list[i]->GetStep();
          version_list->push_back(dump_version);
        } else {
          version_list->push_back(0);
        }
      }
    }
    return key_list->size();
  }

  Status Shrink(int64 gs) {
    if (emb_config_.steps_to_live > 0) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>* > value_ptr_list;
      kv_->GetSnapshot(&key_list, &value_ptr_list);
      std::vector<std::pair<K, ValuePtr<V>* > > to_deleted;
      for (int64 i = 0; i < key_list.size(); ++i) {
        int64 version = value_ptr_list[i]->GetStep();
        if (gs - version > emb_config_.steps_to_live) {
          to_deleted.push_back(std::pair<K, ValuePtr<V>*>(key_list[i], value_ptr_list[i]));
        }
      }
      for (const auto it : to_deleted) {
        // TODO memory recycle
        (it.second)->Destroy(value_len_);
        delete it.second;
        kv_->Remove(it.first);
      }
    }
    return Status::OK();
  }

  Status Destroy(int64 value_len) {
    std::vector<K> key_list;
    std::vector<ValuePtr<V>* > value_ptr_list;
    kv_->GetSnapshot(&key_list, &value_ptr_list);
    for (auto value_ptr : value_ptr_list) {
      value_ptr->Destroy(value_len);
      delete value_ptr;
    }
    return Status::OK();
  }

  mutex* mu() {
    return &mu_;
  }

  KVInterface<K, V>* kv() {
    return kv_;
  }

 private:
  std::string name_;
  KVInterface<K, V>* kv_;
  bool is_initialized_ = false;

  mutex mu_;

  V* default_value_;
  int64 value_len_;
  Allocator* alloc_;
  EmbeddingConfig emb_config_;

  ~EmbeddingVar() override {
    if (emb_config_.is_primary()) {
      Destroy(value_len_);
      delete kv_;
    }
    TypedAllocator::Deallocate(alloc_, default_value_, value_len_);
  }
  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVar);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_



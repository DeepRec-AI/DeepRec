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

#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/framework/embedding/embedding_filter.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"

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
               KVInterface<K, V>* kv,
               EmbeddingConfig emb_cfg = EmbeddingConfig()):
      name_(name),
      kv_(kv),
      default_value_(nullptr),
      value_len_(0),
      alloc_(nullptr),
      emb_config_(emb_cfg) {}

  Status Init() {
    if (kv_ == nullptr) {
       return errors::InvalidArgument("Invalid ht_type to construct EmbeddingVar");
    } else {
      return Status::OK();
    }
  }

  Status Init(const Tensor& default_tensor) {
    if (LayoutType::LIGHT == emb_config_.get_layout_type()) {
      new_value_ptr_fn = [] (size_t size) { return new LightValuePtr<V>(size); };
    } else if (LayoutType::NORMAL == emb_config_.get_layout_type()) {
      new_value_ptr_fn = [] (size_t size) { return new NormalValuePtr<V>(size); };
    } else {
      return errors::InvalidArgument(name_, ", Unsupport EmbeddingVariable LayoutType.");
    }
    filter_ = FilterFactory::CreateFilter<K, V, EmbeddingVar<K, V>>(emb_config_, this);

    if (embedding::StorageType::DRAM == emb_config_.get_storage_type()) {
      alloc_ = ev_allocator();
      if (!alloc_) {
        return errors::InvalidArgument(name_, ", No registered EV AllocatorFactory.");
      }
    } else if (embedding::StorageType::PMEM_MEMKIND == emb_config_.get_storage_type()) {
      alloc_ = pmem_allocator();
      if (!alloc_) {
        return errors::InvalidArgument(name_, ", No registered PMEM_MEMKIND AllocatorFactory.");
      }
    } else if (embedding::StorageType::PMEM_LIBPMEM == emb_config_.get_storage_type()){
      alloc_ = experimental_pmem_allocator(emb_config_.get_storage_path(), emb_config_.get_storage_size());
      if (!alloc_) {
        return errors::InvalidArgument(name_, ", No registered PMEM_LIBPMEM AllocatorFactory.");
      }
    } else {
      return errors::InvalidArgument(name_, ", Unsupport EmbeddingVariable StorageType.");
    }

    if (kv_ == nullptr) {
       return errors::InvalidArgument("Invalid ht_type to construct EmbeddingVar");
    } else {
      value_len_ = default_tensor.NumElements();
      default_value_ = TypedAllocator::Allocate<V>(alloc_, default_tensor.NumElements(), AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      memcpy(default_value_, &default_tensor_flat(0), default_tensor.TotalBytes());
      return Status::OK();
    }
  }

  Status Init(const Tensor& default_tensor, int64 default_value_dim) {
    if (LayoutType::LIGHT == emb_config_.get_layout_type()) {
      new_value_ptr_fn = [] (size_t size) { return new LightValuePtr<V>(size); };
    } else if (LayoutType::NORMAL == emb_config_.get_layout_type()) {
      new_value_ptr_fn = [] (size_t size) { return new NormalValuePtr<V>(size); };
    } else if (LayoutType::LEVELDB == emb_config_.get_layout_type()) {
     if (emb_config_.is_primary()) {
      std::string path = emb_config_.get_storage_path();
      Status s = Env::Default()->IsDirectory(path);
      if (!s.ok()) {
        LOG(WARNING) << "StoragePath=\"" << path << "\" is not Directory, message: " << s.ToString() << ". Try to create dir.";
        TF_CHECK_OK(Env::Default()->RecursivelyCreateDir(path));
      }
      db_name_ = io::JoinPath(path, "level_db_" + std::to_string(Env::Default()->NowMicros()));
      leveldb::Status st;
      leveldb::Options options;
      options.create_if_missing = true;
      //options.write_buffer_size = 1024 * 1024 * 1024;
      //options.error_if_exists = true;
      st = leveldb::DB::Open(options, db_name_.c_str(), &level_db_);
      if (!st.ok()) {
        LOG(FATAL) << "Fail to open leveldb: " << st.ToString();
      } else {
        VLOG(1) << "Open DB Success, db_name: " << db_name_;
      }
      new_value_ptr_fn = [this] (size_t size) { return new DBValuePtr<V>(size, this->level_db_); };
     }
    } else {
      return errors::InvalidArgument(name_, ", Unsupport EmbeddingVariable LayoutType.");
    }
    filter_ = FilterFactory::CreateFilter<K, V, EmbeddingVar<K, V>>(emb_config_, this);

    if (embedding::StorageType::DRAM == emb_config_.get_storage_type()) {
      alloc_ = ev_allocator();
      if (!alloc_) {
        return errors::InvalidArgument(name_, ", No registered EV AllocatorFactory.");
      }
    } else if (embedding::StorageType::PMEM_MEMKIND == emb_config_.get_storage_type()) {
      alloc_ = pmem_allocator();
      if (!alloc_) {
        return errors::InvalidArgument(name_, ", No registered PMEM_MEMKIND AllocatorFactory.");
      }
    } else if (embedding::StorageType::PMEM_LIBPMEM == emb_config_.get_storage_type()){
      alloc_ = experimental_pmem_allocator(emb_config_.get_storage_path(), emb_config_.get_storage_size());
      if (!alloc_) {
        return errors::InvalidArgument(name_, ", No registered PMEM_LIBPMEM AllocatorFactory.");
      }
    } else if (embedding::StorageType::LEVELDB == emb_config_.get_storage_type()) {
      alloc_ = ev_allocator();
      if (!alloc_) {
        return errors::InvalidArgument(name_, ", No registered EV AllocatorFactory.");
      }
    } else {
      return errors::InvalidArgument(name_, ", Unsupport EmbeddingVariable StorageType.");
    }
    
    if (kv_ == nullptr) {
       return errors::InvalidArgument("Invalid ht_type to construct EmbeddingVar");
    } else {
      emb_config_.default_value_dim = default_value_dim;
      value_len_ = default_tensor.NumElements()/emb_config_.default_value_dim;
      default_value_ = TypedAllocator::Allocate<V>(cpu_allocator(), default_tensor.NumElements(), AllocationAttributes());
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

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr, bool* is_filter,
      int64 update_version = -1) {
    return filter_->LookupOrCreateKey(key, value_ptr, is_filter, update_version);
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

  int64 GetFreq(K key) {
    return filter_->GetFreq(key);
  }

  void LookupOrCreate(K key, V* val, V* default_v)  {
    const V* default_value_ptr = (default_v == nullptr) ? default_value_ : default_v;
    filter_->LookupOrCreate(key, val, default_value_ptr);
  }

  void LookupOrCreate(K key, V* val, V* default_v, int64 count)  {
    const V* default_value_ptr = (default_v == nullptr) ? default_value_ : default_v;
    filter_->LookupOrCreate(key, val, default_value_ptr, count);
  }

  V* LookupOrCreateEmb(ValuePtr<V>* value_ptr, const V* default_v) {
    return value_ptr->GetOrAllocate(alloc_, value_len_, default_v,
        emb_config_.emb_index);
  }

  V* LookupPrimaryEmb(ValuePtr<V>* value_ptr) {
    V* primary_val = value_ptr->GetValue(emb_config_.primary_emb_index, value_len_);
    return primary_val;
  }

  typename TTypes<V>::Flat flat(ValuePtr<V>* value_ptr) {
    V* val = LookupOrCreateEmb(value_ptr, default_value_);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }

  void Commit(ValuePtr<V>* value_ptr, const V* v) {
    value_ptr->Commit(value_len_, v, emb_config_.emb_index);
  }

  int64 ValueLen() const {
    return value_len_;
  }

  int64 Size() const {
    return kv_->Size();
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
      value_ptr->Free(v);
    }
    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list,
                    std::vector<int64>* version_list, std::vector<int64>* freq_list) {
    std::vector<ValuePtr<V>* > value_ptr_list;
    std::vector<K> key_list_tmp;
    kv_->GetSnapshot(&key_list_tmp, &value_ptr_list);
    for (int64 i = 0; i < key_list_tmp.size(); ++i) {
      V* val = value_ptr_list[i]->GetValue(emb_config_.emb_index, value_len_);
      V* primary_val = value_ptr_list[i]->GetValue(emb_config_.primary_emb_index, value_len_);
      if (val != nullptr && primary_val != nullptr) {
        value_list->push_back(val);
        key_list->push_back(key_list_tmp[i]);
        if (emb_config_.filter_freq != 0) {
          int64 dump_freq = filter_->GetFreq(key_list_tmp[i], value_ptr_list[i]);
          freq_list->push_back(dump_freq); 
        }
        if (emb_config_.steps_to_live != 0) {
          int64 dump_version = value_ptr_list[i]->GetStep();
          version_list->push_back(dump_version);
        }
      }
    }
    return key_list->size();
  }

  Status Destroy(int64 value_len) {
    std::vector<K> key_list;
    std::vector<ValuePtr<V>* > value_ptr_list;
    kv_->GetSnapshot(&key_list, &value_ptr_list);
    for (auto value_ptr : value_ptr_list) {
      value_ptr->Destroy(alloc_, value_len);
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

  Status Shrink() {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>* > value_ptr_list;
      kv_->GetSnapshot(&key_list, &value_ptr_list);
      std::vector<std::pair<K, ValuePtr<V>* > > to_deleted;
      for (int64 i = 0; i < key_list.size(); ++i) {
        V* val = LookupPrimaryEmb(value_ptr_list[i]);
        V l2_weight = 0.0;
        for (int64 j = 0; j < value_len_; j++) {
            l2_weight += val[j] * val[j];
        }
        l2_weight *= 0.5;
        if (l2_weight < emb_config_.l2_weight_threshold) {
          to_deleted.push_back(std::pair<K, ValuePtr<V>*>(key_list[i], value_ptr_list[i]));
        }
        value_ptr_list[i]->Free(val);
      }
      for (const auto it : to_deleted) {
        // TODO memory recycle
        //(it.second)->Destroy(value_len_);
        //delete it.second;
        kv_->Remove(it.first);
      }
    return Status::OK();
  }

  Status Shrink(int64 gs) {
    if (emb_config_.steps_to_live > 0) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>* > value_ptr_list;
      kv_->GetSnapshot(&key_list, &value_ptr_list);
      std::vector<std::pair<K, ValuePtr<V>* > > to_deleted;
      for (int64 i = 0; i < key_list.size(); ++i) {
        int64 version = value_ptr_list[i]->GetStep();
        if (version == -1) {
          value_ptr_list[i]->SetStep(gs);
        } else {
          if (gs - version > emb_config_.steps_to_live) {
            to_deleted.emplace_back(std::pair<K, ValuePtr<V>*>(key_list[i], value_ptr_list[i]));
          }
        }
      }
      for (const auto it : to_deleted) {
        // TODO memory recycle
        (it.second)->Destroy(alloc_, value_len_);
        delete it.second;
        kv_->Remove(it.first);
      }
    }
    return Status::OK();
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

 private:
  Status LookupOrCreateKeyInternal(K key, ValuePtr<V>** value_ptr, size_t size) {
    Status s = kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      // Found
      return s;
    } else {
      // Not found
      *value_ptr = new_value_ptr_fn(size);
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

 private:
  std::string name_;
  KVInterface<K, V>* kv_;
  bool is_initialized_ = false;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn;

  mutex mu_;

  V* default_value_;
  int64 value_len_;
  Allocator* alloc_;
  EmbeddingConfig emb_config_;
  EmbeddingFilter<K, V, EmbeddingVar<K, V>>* filter_;
  leveldb::DB* level_db_;
  std::string db_name_;

  ~EmbeddingVar() override {
    if (emb_config_.is_primary()) {
      if (LayoutType::LEVELDB == emb_config_.get_layout_type()) {
        delete level_db_;
        int64 undeleted_files = 0;
        int64 undeleted_dirs = 0;
        TF_CHECK_OK(Env::Default()->DeleteRecursively(db_name_, &undeleted_files, &undeleted_dirs));
      }
      Destroy(value_len_);
      delete kv_;
    }
    TypedAllocator::Deallocate(alloc_, default_value_, value_len_);
  }
  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVar);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_



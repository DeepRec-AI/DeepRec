/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_INCR_SAVE_RESTORE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_INCR_SAVE_RESTORE_OPS_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
template <typename T>
class ThreadSafeHashMap {
 public:
  ThreadSafeHashMap() {}
  ~ThreadSafeHashMap() {}

 public:
  void Update(const Tensor& indices, int64 start, int64 end) {
    mutex_lock l(lock_);
    auto indices_flat = indices.flat<T>();
    for (int64 idx = start; idx < end; idx++) {
      auto indice = indices_flat(idx);
      auto it = hash_map_.find(indice);
      if (it == hash_map_.end()) {
        hash_map_[indice] = 1;
      } else {
        it->second = it->second + 1;
      }
    }
  }

  void Swap(std::unordered_map<T, uint64>& out) {
    mutex_lock l(lock_);
    hash_map_.swap(out);
  }

  void GetKeys(std::set<T>& key_set) {
    mutex_lock l(lock_);
    for (auto it : hash_map_) {
      key_set.insert(it.first);
    }
  }

  void Clear() {
    mutex_lock l(lock_);
    hash_map_.clear();
  }

 private:
  std::unordered_map<T, uint64> hash_map_;
  mutex lock_;
};

template <typename T>
class ParallelHashMap {
 public:
  explicit ParallelHashMap(int min_part_size = 128, int part_count = 32)
      : part_count_(part_count),
      min_part_size_(min_part_size) {
    hash_maps_.resize(part_count_);
  }

  void Update(const Tensor& indices, OpKernelContext *ctx) {
    const int64 N = indices.NumElements();
    auto thread_pool = *(ctx->device()->tensorflow_cpu_worker_threads());

    std::vector<std::pair<int64, int64>> parts;
    SplitParallelParts(N,
        std::min(part_count_, thread_pool.workers->NumThreads()), parts);

    int part_count = parts.size();
    BlockingCounter counter(part_count);
    for (int i = 0; i < part_count; i++) {
      int64 start = parts[i].first;
      int64 end = parts[i].second;
      thread_pool.workers->Schedule([this, indices, i, start, end, &counter]() {
          hash_maps_[i].Update(indices, start, end);
          counter.DecrementCount();
        });
    }
    counter.Wait();
  }

  void Swap(std::unordered_map<T, uint64> &indices) {
    std::vector<std::unordered_map<T, uint64>> tmp_maps;
    tmp_maps.resize(part_count_);
    for (int i = 0; i < part_count_; i++) {
      hash_maps_[i].Swap(tmp_maps[i]);
    }

    indices.clear();
    for (int i = 0; i < part_count_; i++) {
      for (auto it : tmp_maps[i]) {
        auto indiceIt = indices.find(it.first);
        if (indiceIt == indices.end()) {
          indices[it.first] = it.second;
        } else {
          indices[it.first] += it.second;
        }
      }
    }
  }

  void Clear() {
    for (size_t i = 0; i < part_count_; i++) {
      hash_maps_[i].Clear();
    }
  }

  void GetKeys(std::set<T>& key_set) {
    for (size_t i = 0; i < part_count_; i++) {
      hash_maps_[i].GetKeys(key_set);
    }
  }

  void SplitParallelParts(int64 total_num, int64 part_count,
      std::vector<std::pair<int64, int64>>& parts) {
    if (total_num == 0) {
      return;
    }

    int64 actual_part_count = part_count;
    int64 part_size = total_num / actual_part_count;
    if (part_size < min_part_size_) {
      actual_part_count = total_num / min_part_size_;
      actual_part_count = actual_part_count == 0 ? 1 : actual_part_count;
    }

    part_size = total_num / actual_part_count;
    int64 left = total_num % actual_part_count;
    int64 start = 0;
    for (int i = 0; i < actual_part_count; i++) {
      int64 end = start + part_size + (left > 0 ? 1 : 0);
      parts.push_back(std::make_pair(start, end));
      start = end;
      left -= 1;
    }
  }

 private:
  std::vector<ThreadSafeHashMap<T> > hash_maps_;
  int part_count_;
  int min_part_size_;
};

template <class K>
class IncrKeyDumpIterator : public DumpIterator<K> {
 public:
  explicit IncrKeyDumpIterator(std::vector<K>& incr_keys)
    : incr_keys_(incr_keys) {
    keys_iter_  = incr_keys_.begin();
  }

  bool HasNext() const {
    return keys_iter_ != incr_keys_.end();
  }

  K Next() {
    return *keys_iter_++;
  }

 private:
  std::vector<K>& incr_keys_;
  typename std::vector<K>::iterator keys_iter_;
};

template<class K, class T>
class IncrEVValueDumpIterator : public  DumpIterator<T> {
 public:
  IncrEVValueDumpIterator(std::vector<K>& incr_keys,
      EmbeddingVar<K, T>*& emb_var)
    : incr_keys_(incr_keys),
    emb_var_(emb_var) {
    keys_iter_ = incr_keys_.begin();
    keys_idx_ = 1;
    col_idx_ = 0;
  }

  bool HasNext() const {
    if (keys_iter_ != incr_keys_.end()) {
      if (keys_idx_ < incr_keys_.size()) {
        return true;
      } else {
        return col_idx_ < emb_var_->ValueLen();
      }
    } else {
      return false;
    }
  }

  T Next() {
    if (col_idx_ >= emb_var_->ValueLen()) {
      keys_iter_++;
      keys_idx_++;
      col_idx_ = 0;
    }
    ValuePtr<T>* value_ptr = NULL;
    TF_CHECK_OK(emb_var_->LookupOrCreateKey(*keys_iter_, &value_ptr));
    return emb_var_->flat(value_ptr, *keys_iter_)(col_idx_++);
  }

 private:
  int64 keys_idx_;
  int64 col_idx_;
  typename std::vector<K>::iterator keys_iter_;
  std::vector<K>& incr_keys_;
  EmbeddingVar<K, T>* emb_var_;
};

template<class K, class V, class T>
class IncrEVVersionDumpIterator : public  DumpIterator<T> {
 public:
  IncrEVVersionDumpIterator(std::vector<K>& incr_keys,
      EmbeddingVar<K, V>*& emb_var) :
    incr_keys_(incr_keys),
    emb_var_(emb_var) {
    keys_iter_ = incr_keys_.begin();
  }

  bool HasNext() const {
    return keys_iter_ != incr_keys_.end();
  }

  T Next() {
    if (emb_var_->StepsToLive() == 0) {
      keys_iter_++;
      return 0;
    } else {
      K key = *keys_iter_;
      int64 dump_version = emb_var_->GetVersion(key);
      keys_iter_++;
      return dump_version;
    }
  }

 private:
  std::vector<K>& incr_keys_;
  typename std::vector<K>::iterator keys_iter_;
  EmbeddingVar<K, V>* emb_var_;
};

template<class K, class V, class T>
class IncrEVFreqDumpIterator : public  DumpIterator<T> {
 public:
  IncrEVFreqDumpIterator(std::vector<K>& incr_keys,
      EmbeddingVar<K, V>*& emb_var)
    : incr_keys_(incr_keys),
    emb_var_(emb_var) {
    keys_iter_ = incr_keys_.begin();
  }

  bool HasNext () const {
    return keys_iter_ != incr_keys_.end();
  }

  T Next() {
    K key = *keys_iter_;
    int64 dump_version = emb_var_->GetFreq(key);
    keys_iter_++;
    return dump_version;
  }

 private:
  std::vector<K>& incr_keys_;
  typename std::vector<K>::iterator keys_iter_;
  EmbeddingVar<K, V>* emb_var_;
};

template<class K, class T>
class IncrNormalValueDumpIterator : public  DumpIterator<T> {
 public:
  IncrNormalValueDumpIterator(std::vector<K>& incr_keys,
      const Tensor& variable)
    : incr_keys_(incr_keys),
    variable_(variable) {
    var_data_ = (T*)variable.flat<T>().data();
    keys_iter_ = incr_keys_.begin();
    keys_idx_ = 1;
    col_idx_ = 0;
  }

  bool HasNext() const {
    if (keys_iter_ != incr_keys_.end()) {
      if (keys_idx_ < incr_keys_.size()) {
        return true;
      } else {
        return col_idx_ < variable_.dim_size(1);
      }
    } else {
      return false;
    }
  }

  T Next() {
    if (col_idx_ >= variable_.dim_size(1)) {
      keys_iter_++;
      keys_idx_++;
      col_idx_ = 0;
    }
    T val = var_data_[(*keys_iter_) * variable_.dim_size(1) + col_idx_];
    col_idx_++;
    return val;
  }

 private:
  std::vector<K>& incr_keys_;
  T* var_data_;
  int64 col_limit_;
  int64 keys_idx_;
  typename std::vector<K>::iterator keys_iter_;
  int64 col_idx_;
  const Tensor& variable_;
};

template <typename K, typename V = float>
class IndicesIncrRecorder: public ResourceBase {
 public:
  explicit IndicesIncrRecorder(const std::string &name,
      int32 part_count = 16, int32 min_part_size = 128)
      : name_(name),
      incr_indices_(min_part_size, part_count) {}

  void UpdateIndices(const Tensor& indices, OpKernelContext *ctx) {
    if (global_version_ == -1) {
      return;
    }

    incr_indices_.Update(indices, ctx);
  }

  void UpdateGlobalVersion() {
    global_version_ = 1;
    mutex_lock l(mu_);
    incr_indices_.Clear();
  }

  void SwapIndices(std::unordered_map<K, uint64>& indices) {
    incr_indices_.Swap(indices);
  }

  Status DumpSparseNormalTensor(const string& tensor_name,
      const Tensor& variable, BundleWriter* writer) {
    mutex_lock l(mu_);
    size_t bytes_limit = 8 << 20;
    char* dump_buffer = (char*)malloc(sizeof(char) * bytes_limit);

    std::set<K> incr_keys_set;
    incr_indices_.GetKeys(incr_keys_set);
    std::vector<K> incr_keys;
    incr_keys.assign(incr_keys_set.begin(), incr_keys_set.end());

    IncrKeyDumpIterator<K> key_dump_iter(incr_keys);
    Status st = SaveTensorWithFixedBuffer(tensor_name + "-sparse_incr_keys",
        writer, dump_buffer, bytes_limit, &key_dump_iter,
        TensorShape({incr_keys.size()}));
    if (!st.ok()) {
      free(dump_buffer);
      return st;
    }

    IncrNormalValueDumpIterator<K, V> value_dump_iter(incr_keys, variable);
    st = SaveTensorWithFixedBuffer(tensor_name + "-sparse_incr_values",
        writer, dump_buffer, bytes_limit,
        &value_dump_iter,
        TensorShape({incr_keys.size(), variable.dim_size(1)}));
    if (!st.ok()) {
      free(dump_buffer);
      return st;
    }

    free(dump_buffer);
    return Status::OK();
  }

  Status DumpSparseEmbeddingTensor(const string& tensor_name,
      EmbeddingVar<K, V>* emb_var, BundleWriter* writer,
      OpKernelContext* context) {
    mutex_lock l(mu_);
    size_t bytes_limit = 8 << 20;
    char* dump_buffer = (char*)malloc(sizeof(char) * bytes_limit);

    std::set<K> incr_keys;
    incr_indices_.GetKeys(incr_keys);

    std::vector<std::vector<K> > incr_keys_parts;
    incr_keys_parts.resize(kSavedPartitionNum);

    for (auto& ik : incr_keys) {
      for (int partid = 0; partid < kSavedPartitionNum; partid++) {
        if (ik % kSavedPartitionNum == partid &&
            emb_var->GetFreq(ik) >= emb_var->MinFreq()) {
          incr_keys_parts[partid].push_back(ik);
          break;
        }
      }
    }

    std::vector<K> partitioned_incr_keys;
    Tensor part_offset_tensor;
    context->allocate_temp(DT_INT32,
        TensorShape({kSavedPartitionNum + 1}), &part_offset_tensor);
    auto part_offset_flat = part_offset_tensor.flat<int32>();
    part_offset_flat(0) = 0;
    int ptsize = 0;
    for (int partid = 0; partid < kSavedPartitionNum; partid++) {
      std::vector<K>& key_list = incr_keys_parts[partid];

      ptsize += key_list.size();
      for (int inpid = 0; inpid < key_list.size(); inpid++) {
        partitioned_incr_keys.push_back(key_list[inpid]);
      }

      part_offset_flat(partid + 1) = part_offset_flat(partid) + key_list.size();
    }
    writer->Add(tensor_name+ "-incr_partition_offset", part_offset_tensor);

    IncrKeyDumpIterator<K> key_dump_iter(partitioned_incr_keys);
    Status st = SaveTensorWithFixedBuffer(tensor_name + "-sparse_incr_keys",
        writer, dump_buffer, bytes_limit, &key_dump_iter,
        TensorShape({partitioned_incr_keys.size()}));
    if (!st.ok()) {
      free(dump_buffer);
      return st;
    }

    IncrEVValueDumpIterator<K, V> ev_value_dump_iter(
        partitioned_incr_keys, emb_var);
    st = SaveTensorWithFixedBuffer(tensor_name + "-sparse_incr_values",
        writer, dump_buffer, bytes_limit, &ev_value_dump_iter,
        TensorShape({(uint64)partitioned_incr_keys.size(),
          emb_var->ValueLen()}));
    if (!st.ok()) {
      free(dump_buffer);
      return st;
    }

    IncrEVVersionDumpIterator<K, V, int64> ev_version_dump_iter(
        partitioned_incr_keys, emb_var);
    st = SaveTensorWithFixedBuffer(tensor_name + "-sparse_incr_versions",
        writer, dump_buffer, bytes_limit, &ev_version_dump_iter,
        TensorShape({(uint64)partitioned_incr_keys.size()}));
    if (!st.ok()) {
      free(dump_buffer);
      return st;
    }
    IncrEVFreqDumpIterator<K, V, int64> ev_freq_dump_iter(
        partitioned_incr_keys, emb_var);
    st = SaveTensorWithFixedBuffer(tensor_name + "-sparse_incr_freqs",
        writer, dump_buffer, bytes_limit, &ev_freq_dump_iter,
        TensorShape({(uint64)partitioned_incr_keys.size()}));
    if (!st.ok()) {
      free(dump_buffer);
      return st;
    }
    free(dump_buffer);
    return Status::OK();
  }

  string DebugString() const {
    return "IndicesIncrRecorder";
  }

  string GetName() {
    return name_;
  }

 private:
  mutex mu_;
  string name_;
  ParallelHashMap<K> incr_indices_;
  std::atomic<int64> global_version_ = {-1};

  TF_DISALLOW_COPY_AND_ASSIGN(IndicesIncrRecorder);
};

class SparsePartitioner {
 public:
  SparsePartitioner(int64 part_count, int64_t part_idx,
                    int64 hash_bucket_size)
      : part_count_(part_count),
      part_idx_(part_idx),
      hash_bucket_size_(hash_bucket_size) {
    assert(part_idx_ >= part_count_);
  }

  virtual int64 CalcGlobalOffset(int64 part_offset) = 0;

  std::string toString() const {
    return strings::Printf(
        "part_mode:%s, part_count:%lld, part_idx:%ld, hash_bucket_size:%ld",
        part_mode_.c_str(), part_count_, part_idx_, (long)hash_bucket_size_);
  }

 protected:
  std::string part_mode_;
  int64 part_count_;
  int64 part_idx_;
  int64 hash_bucket_size_;
};

class DivSparsePartitioner : public SparsePartitioner {
 public:
  DivSparsePartitioner(int64 part_count, int64 part_idx,
                       int64 hash_bucket_size)
      : SparsePartitioner(part_count, part_idx, hash_bucket_size) {
    part_mode_ = "div";
    int64 ids_per_part = hash_bucket_size_ / part_count_;
    int64 extras = hash_bucket_size_ % part_count_;

    part_offset_start_ = 0;
    for (int i = 0; i < part_idx; i++) {
      part_offset_start_ += (i < extras ? (ids_per_part + 1) : ids_per_part);
    }
  }

  int64 CalcGlobalOffset(int64 part_offset) {
    return part_offset_start_ + part_offset;
  }
 private:
  int64 part_offset_start_;
};

class ModSparsePartitioner : public SparsePartitioner {
 public:
  ModSparsePartitioner(int64 part_count, int64 part_idx,
                       int64 hash_bucket_size)
      : SparsePartitioner(part_count, part_idx, hash_bucket_size) {
    part_mode_ = "mod";
  }

  int64 CalcGlobalOffset(int64 part_offset) {
    return part_offset * part_count_ + part_idx_;
  }
};

} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_INCR_SAVE_RESTORE_OPS_H_

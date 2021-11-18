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

#ifndef TENSORFLOW_CORE_FRAMEWORK_HASH_TABLE_TENSIBLE_VARIABLE_H_
#define TENSORFLOW_CORE_FRAMEWORK_HASH_TABLE_TENSIBLE_VARIABLE_H_

#include <vector>
#include <deque>
#include <memory>
#include <string>

#include "tensorflow/core/framework/hash_table/tensor_generator.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

class TensibleVariable : public core::RefCounted {
 public:
  TensibleVariable(
      TensorGenerator* generator, const TensorShape& shape, DataType dtype);
  ~TensibleVariable();
  void Resize(int64 size, const std::function<void(Status)>& done);
  int64 SegmentSize() const { return segment_size_; }
  int64 Size() const { return size_; }
  const TensorShape& shape() const {return shape_; }
  DataType dtype() const {return dtype_; }
  int64 SliceSize() const { return slice_size_; }
  template<typename T = void>
  T* GetSlice(int64_t id) const {
    return reinterpret_cast<T*>
      (ptrs_[id / segment_size_] + (id % segment_size_) * slice_size_);
  }

  void LockUpdate() {
    return update_mu_.lock();
  }

  void UnlockUpdate() {
    return update_mu_.unlock();
  }

  void Clear();
  void ClearIds(int64* ids, int64 size, const std::function<void(Status)>& done);

  // Only For "Load From Checkpoint"
  void ZeroCostResize(int64 size);
  void Pad(int64 size, const std::function<void(Status)>& done);

  mutex* GetRWLock() {
    return &rwlock_;
  }

  TensorGenerator* GetGenerator() const {
    return generator_;
  }
  
 private:
  TensorGenerator* generator_;
  TensorShape shape_;
  DataType dtype_;
  mutable mutex rwlock_;

  Eigen::array<Eigen::DenseIndex, 1> eigen_slice_shape_;
  int64 segment_size_;
  int64 slice_size_;
  std::atomic<int64> size_;

  mutex structure_update_mu_;

  std::vector<Tensor> tensors_;

  struct PtrSpec {
    std::unique_ptr<char*[]> ptr;
    size_t size;
  };
  std::vector<PtrSpec> all_ptr_vec_;

  char** ptrs_;

  mutex update_mu_;

  static constexpr int kPtrStartSize = 4;
};

class TensibleVariableResource : public ResourceBase {
 public:
  TensibleVariableResource() : internal_(nullptr), initialized_(false) { }

  ~TensibleVariableResource() {
    if (internal_ != nullptr) {
      internal_->Unref();
    }
  }

  string DebugString() const override {
    return "SimpleHashTable";
  }

  Status CreateInternal(
      TensorGenerator* generator, const TensorShape& shape, DataType dtype) {
    mutex_lock lock(init_mu_);
    initialized_ = false;
    if (internal_ != nullptr) {
      return errors::FailedPrecondition("HashTable has been initialized");
    }
    internal_ = new TensibleVariable(generator, shape, dtype);
    return Status::OK();
  }

  TensibleVariable* Internal() {
    return internal_;
  }

  bool Initialized() {
    mutex_lock lock(init_mu_);
    return initialized_;
  }

  void SetInitialized(bool initialized) {
    mutex_lock lock(init_mu_);
    initialized_ = initialized;
  }

 private:
  mutex init_mu_;
  TensibleVariable* internal_;
  bool initialized_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_HASH_TABLE_TENSIBLE_VARIABLE_H_

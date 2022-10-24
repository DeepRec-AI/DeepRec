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

#include "tensorflow/core/framework/hash_table/tensible_variable.h"
#include "tensorflow/core/framework/hash_table/status_collector.h"

#include <future>

namespace tensorflow {

TensibleVariable::TensibleVariable(
    TensorGenerator* generator, const TensorShape& shape, DataType dtype)
    : generator_(generator), shape_(shape), dtype_(dtype) {
  size_.store(0);
  slice_size_ = 1;
  for (int i = 1; i < shape_.dims(); i++) {
    slice_size_ *= shape_.dim_size(i);
  }
  eigen_slice_shape_[0] = slice_size_;
  slice_size_ *= DataTypeSize(dtype);
  segment_size_ = shape_.dim_size(0);
  all_ptr_vec_.emplace_back();
  all_ptr_vec_.back().size = kPtrStartSize;
  all_ptr_vec_.back().ptr.reset(new char*[kPtrStartSize]);
  ptrs_ = all_ptr_vec_.back().ptr.get();
  generator_->Ref();
}

TensibleVariable::~TensibleVariable() {
  generator_->Unref();
}

void TensibleVariable::Resize(
    int64 size, const std::function<void(Status)>& done) {
  if (size <= size_) {
    done(Status::OK());
    return;
  }
  int64_t segment_count = (size - size_ - 1) / segment_size_ + 1;
  StatusCollector* stc = new StatusCollector(segment_count, [this, done](Status st) {
    if (st.ok()) {
      ptrs_ = all_ptr_vec_.back().ptr.get();
      size_ = tensors_.size() * segment_size_;
    }
    done(st);
  });
  for (int i = 0; i < segment_count; i++) {
    generator_->GetNextTensor([this, stc] (Status st, const Tensor& tensor) {
      auto fn = [&]() -> Status {
        mutex_lock lock(structure_update_mu_);
        if (!st.ok()) {
          return st;
        }
        if (tensor.shape() != shape_) {
          return errors::InvalidArgument(
              "Tensor Generator generate shape error ",
              tensor.shape().DebugString(), " vs ", shape_.DebugString());
        }
        if (tensor.dtype() != dtype_) {
          return errors::InvalidArgument(
              "Tensor Generator generate dtype error ",
              tensor.dtype(), " vs ", dtype_);
        }
        tensors_.push_back(tensor);
        if (all_ptr_vec_.back().size < tensors_.size()) {
          all_ptr_vec_.emplace_back();
          auto&& old_spec = all_ptr_vec_[all_ptr_vec_.size() - 2];
          auto&& new_spec = all_ptr_vec_[all_ptr_vec_.size() - 1];
          new_spec.size = old_spec.size * 2;
          new_spec.ptr.reset(new char*[new_spec.size]);
          memcpy(new_spec.ptr.get(), old_spec.ptr.get(),
              old_spec.size * sizeof(char*));
        }
        all_ptr_vec_.back().ptr[tensors_.size() - 1] =
          const_cast<char*>(tensor.tensor_data().data());
        return Status::OK();
      };
      stc->AddStatus(fn());
    });
  }
  stc->Start();
}

void TensibleVariable::ZeroCostResize(int64 size) {
  mutex_lock lock(structure_update_mu_);
  size_t segment_count = (size + segment_size_ - 1) / segment_size_;
  while (tensors_.size() < segment_count) {
    tensors_.emplace_back(dtype_, shape_);
    if (all_ptr_vec_.back().size < tensors_.size()) {
      all_ptr_vec_.emplace_back();
      auto&& old_spec = all_ptr_vec_[all_ptr_vec_.size() - 2];
      auto&& new_spec = all_ptr_vec_[all_ptr_vec_.size() - 1];
      new_spec.size = old_spec.size * 2;
      new_spec.ptr.reset(new char*[new_spec.size]);
      memcpy(new_spec.ptr.get(), old_spec.ptr.get(),
          old_spec.size * sizeof(char*));
    }
    all_ptr_vec_.back().ptr[tensors_.size() - 1] =
      const_cast<char*>(tensors_.back().tensor_data().data());
  }
  size_ = tensors_.size() * segment_size_;
  ptrs_ = all_ptr_vec_.back().ptr.get();
}

void TensibleVariable::Pad(
    int64 size, const std::function<void(Status)>& done) {
  if (size == size_) {
    done(Status::OK());
    return;
  }
  if (size_ - size > segment_size_ || size_ < size) {
    done(errors::FailedPrecondition(
          "TensibleVariable Pad size should less than 1 segment ",
          std::to_string(size_), " ", std::to_string(size)));
    return;
  }
  generator_->GetNextTensor([this, size, done](Status st, const Tensor& tensor) {
    if (!st.ok()) {
      done(st);
      return;
    }
    if (size_ - size > segment_size_ || size_ < size) {
      done(errors::FailedPrecondition(
          "TensibleVariable Pad size should less than 1 segment ",
          std::to_string(size_), " ", std::to_string(size)));
      return;
    }
    int64 pad = size_ - size;
    memcpy(const_cast<char*>(
        tensors_.back().tensor_data().data()) +
          (segment_size_ - pad) * slice_size_,
        tensor.tensor_data().data(), pad * slice_size_);
    done(Status::OK());
  });
}

void TensibleVariable::Clear() {
  mutex_lock lock(structure_update_mu_);
  tensors_.clear();
  size_ = 0;
}

void TensibleVariable::ClearIds(
    int64* ids, int64 size, 
    const std::function<void(Status)>& done) {
  int64 num_segs = (size + segment_size_ - 1) / segment_size_;
  StatusCollector* stc = new StatusCollector(num_segs, done);
  for (int64 i = 0; i < num_segs; ++i) {
    generator_->GetNextTensor([this, stc, ids, i, size] (Status st, const Tensor& tensor) {
      if (!st.ok()) {
        stc->AddStatus(st);
        return;
      }
      {
        mutex_lock wlock(rwlock_);
        int64 k = i * segment_size_;
        int64 sz = std::min(size, (i + 1) * segment_size_);
        while (k < sz) {
          if (ids[k] < size_) {
            char* init_ptr = const_cast<char*>(tensor.tensor_data().data());
            memcpy(GetSlice(ids[k]), init_ptr + (k % segment_size_) * slice_size_, slice_size_);
          }
          ++k;
        }
      }
      stc->AddStatus(Status::OK());
    });
  }
  stc->Start();
}

}  // namespace tensorflow

/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_DUMP_ITERATOR_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_DUMP_ITERATOR_
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
namespace tensorflow {
template <class T>
class DumpIterator;

namespace embedding {
template<class T>
class EVVectorDataDumpIterator: public DumpIterator<T> {
 public:
  EVVectorDataDumpIterator(const std::vector<T>& item_list)
      : curr_iter_(item_list.begin()),
        end_iter_(item_list.end()) {}

  bool HasNext() const {
    return curr_iter_ != end_iter_;
  }

  T Next() {
    T val = *curr_iter_;
    curr_iter_++;
    return val;
  }

 private:
  typename std::vector<T>::const_iterator curr_iter_;
  typename std::vector<T>::const_iterator end_iter_;
};

template<class T>
class EV2dVectorDataDumpIterator: public DumpIterator<T> {
 public:
  EV2dVectorDataDumpIterator(
      std::vector<T*>& valueptr_list,
      int64 value_len,
      ValueIterator<T>* val_iter)
      : curr_iter_(valueptr_list.begin()),
        end_iter_(valueptr_list.end()),
        val_iter_(val_iter),
        value_len_(value_len),
        col_idx_(0) {
    if (!valueptr_list.empty()) {
      if ((int64)*curr_iter_ == ValuePtrStatus::NOT_IN_DRAM) {
        curr_ptr_ = val_iter_->Next();
      } else {
        curr_ptr_ = *curr_iter_;
      }
    }
  }

  bool HasNext() const {
    return curr_iter_ != end_iter_;
  }

  T Next() {
    T val = curr_ptr_[col_idx_++];
    if (col_idx_ >= value_len_) {
      curr_iter_++;
      col_idx_ = 0;
      if (curr_iter_ != end_iter_) {
        if ((int64)*curr_iter_ == ValuePtrStatus::NOT_IN_DRAM) {
          curr_ptr_ = val_iter_->Next();
        } else {
          curr_ptr_ = *curr_iter_;
        }
      }
    }
    return val;
  }

 private:
  typename std::vector<T*>::const_iterator curr_iter_;
  typename std::vector<T*>::const_iterator end_iter_;
  ValueIterator<T>* val_iter_;
  int64 value_len_;
  int64 col_idx_;
  T* curr_ptr_ = nullptr;
};
} //namespace embedding
} //namespace tensorflow
#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_DUMP_ITERATOR_

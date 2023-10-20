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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_CKPT_DATA_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_CKPT_DATA_
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/embedding_var_dump_iterator.h"
namespace tensorflow {
class BundleWriter;
namespace {
  const int kDramFlagOffset = 49;
}

namespace embedding {
template<class K, class V>
class  EmbeddingVarCkptData {
 public:
  void Emplace(K key, void* value_ptr,
               const EmbeddingConfig& emb_config,
               V* default_value,
               FeatureDescriptor<V>* feat_desc,
               bool is_save_freq,
               bool is_save_version,
               bool save_unfiltered_features);

  void Emplace(K key, V* value_ptr);

  void SetWithPartition(
      std::vector<EmbeddingVarCkptData<K, V>>& ev_ckpt_data_parts);

  Status ExportToCkpt(const string& tensor_name,
                      BundleWriter* writer,
                      int64 value_len,
                      ValueIterator<V>* value_iter = nullptr);
 private:
  std::vector<K> key_vec_;
  std::vector<V*> value_ptr_vec_;
  std::vector<int64> version_vec_;
  std::vector<int64> freq_vec_;
  std::vector<K> key_filter_vec_;
  std::vector<int64> version_filter_vec_;
  std::vector<int64> freq_filter_vec_;
  std::vector<int32> part_offset_;
  std::vector<int32> part_filter_offset_;
  const int kSavedPartitionNum = 1000;
};
} //namespace embedding
} //namespace tensorflow
#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_CKPT_DATA_

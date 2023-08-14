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
#include "tensorflow/core/framework/embedding/embedding_var_ckpt_data.h"
#include "tensorflow/core/framework/embedding/embedding_var_dump_iterator.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace embedding {
template<class K, class V>
void EmbeddingVarCkptData<K, V>::Emplace(
    K key, void* value_ptr,
    const EmbeddingConfig& emb_config,
    V* default_value,
    FeatureDescriptor<V>* feat_desc,
    bool is_save_freq,
    bool is_save_version,
    bool save_unfiltered_features) {
  if((int64)value_ptr == ValuePtrStatus::IS_DELETED)
    return;

  bool is_in_dram = ((int64)value_ptr >> kDramFlagOffset == 0);
  bool is_admit = feat_desc->IsAdmit(value_ptr);

  if (is_admit) {
    key_vec_.emplace_back(key);

    if (!is_in_dram) {
      value_ptr_vec_.emplace_back((V*)ValuePtrStatus::NOT_IN_DRAM);
      value_ptr = (void*)((int64)value_ptr & ((1L << kDramFlagOffset) - 1));
    } else if (feat_desc->GetEmbedding(value_ptr, 0) == nullptr) {
      value_ptr_vec_.emplace_back(default_value);
    } else {
      V* val = feat_desc->GetEmbedding(value_ptr, emb_config.emb_index);
      value_ptr_vec_.emplace_back(val);
    }
    if(is_save_version) {
      int64 dump_version = feat_desc->GetVersion(value_ptr);
      version_vec_.emplace_back(dump_version);
    }

    if(is_save_freq) {
      int64 dump_freq = feat_desc->GetFreq(value_ptr);
      freq_vec_.emplace_back(dump_freq);
    }
  } else {
    if (!save_unfiltered_features)
      return;

    key_filter_vec_.emplace_back(key);

    if(is_save_version) {
      int64 dump_version = feat_desc->GetVersion(value_ptr);
      version_filter_vec_.emplace_back(dump_version);
    }

    int64 dump_freq = feat_desc->GetFreq(value_ptr);
    freq_filter_vec_.emplace_back(dump_freq);
  }
}
#define REGISTER_KERNELS(ktype, vtype)                               \
  template void EmbeddingVarCkptData<ktype, vtype>::Emplace(  \
      ktype, void*, const EmbeddingConfig&, \
      vtype*, FeatureDescriptor<vtype>*, bool, bool, bool);
#define REGISTER_KERNELS_ALL_INDEX(type)                             \
  REGISTER_KERNELS(int32, type)                                      \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS


template<class K, class V>
void EmbeddingVarCkptData<K, V>::Emplace(K key, V* value_ptr) {
  key_vec_.emplace_back(key);
  value_ptr_vec_.emplace_back(value_ptr);
}
#define REGISTER_KERNELS(ktype, vtype)                               \
  template void EmbeddingVarCkptData<ktype, vtype>::Emplace(  \
      ktype, vtype*); 
#define REGISTER_KERNELS_ALL_INDEX(type)                             \
  REGISTER_KERNELS(int32, type)                                      \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

template<class K, class V>
void EmbeddingVarCkptData<K, V>::SetWithPartition(
    std::vector<EmbeddingVarCkptData<K, V>>& ev_ckpt_data_parts) {
  part_offset_.resize(kSavedPartitionNum + 1);
  part_filter_offset_.resize(kSavedPartitionNum + 1);
  part_offset_[0] = 0;
  part_filter_offset_[0] = 0;
  for (int i = 0; i < kSavedPartitionNum; i++) {
    part_offset_[i + 1] =
        part_offset_[i] + ev_ckpt_data_parts[i].key_vec_.size();

    part_filter_offset_[i + 1] =
        part_filter_offset_[i] +
        ev_ckpt_data_parts[i].key_filter_vec_.size();

    for (int64 j = 0; j < ev_ckpt_data_parts[i].key_vec_.size(); j++) {
      key_vec_.emplace_back(ev_ckpt_data_parts[i].key_vec_[j]);
    }

    for (int64 j = 0; j < ev_ckpt_data_parts[i].value_ptr_vec_.size(); j++) {
      value_ptr_vec_.emplace_back(ev_ckpt_data_parts[i].value_ptr_vec_[j]);
    }

    for (int64 j = 0; j < ev_ckpt_data_parts[i].version_vec_.size(); j++) {
      version_vec_.emplace_back(ev_ckpt_data_parts[i].version_vec_[j]);
    }

    for (int64 j = 0; j < ev_ckpt_data_parts[i].freq_vec_.size(); j++) {
      freq_vec_.emplace_back(ev_ckpt_data_parts[i].freq_vec_[j]);
    }

    for (int64 j = 0; j < ev_ckpt_data_parts[i].key_filter_vec_.size(); j++) {
      key_filter_vec_.emplace_back(ev_ckpt_data_parts[i].key_filter_vec_[j]);
    }

    for (int64 j = 0; j < ev_ckpt_data_parts[i].version_filter_vec_.size(); j++) {
      version_filter_vec_.emplace_back(ev_ckpt_data_parts[i].version_filter_vec_[j]);
    }

    for (int64 j = 0; j < ev_ckpt_data_parts[i].freq_filter_vec_.size(); j++) {
      freq_filter_vec_.emplace_back(ev_ckpt_data_parts[i].freq_filter_vec_[j]);
    }
  }
}

#define REGISTER_KERNELS(ktype, vtype)                               \
  template void EmbeddingVarCkptData<ktype, vtype>::SetWithPartition(  \
      std::vector<EmbeddingVarCkptData<ktype, vtype>>&); 
#define REGISTER_KERNELS_ALL_INDEX(type)                             \
  REGISTER_KERNELS(int32, type)                                      \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

template<class K, class V>
Status EmbeddingVarCkptData<K, V>::ExportToCkpt(
    const string& tensor_name,
    BundleWriter* writer,
    int64 value_len,
    ValueIterator<V>* value_iter) {
  size_t bytes_limit = 8 << 20;
  std::unique_ptr<char[]> dump_buffer(new char[bytes_limit]);

  EVVectorDataDumpIterator<K> key_dump_iter(key_vec_);
  Status s = SaveTensorWithFixedBuffer(
      tensor_name + "-keys", writer, dump_buffer.get(),
      bytes_limit, &key_dump_iter,
      TensorShape({key_vec_.size()}));
  if (!s.ok())
    return s;

  EV2dVectorDataDumpIterator<V> value_dump_iter(
      value_ptr_vec_, value_len, value_iter);
  s = SaveTensorWithFixedBuffer(
      tensor_name + "-values", writer, dump_buffer.get(),
      bytes_limit, &value_dump_iter,
      TensorShape({value_ptr_vec_.size(), value_len}));
  if (!s.ok())
    return s;

  EVVectorDataDumpIterator<int64> version_dump_iter(version_vec_);
  s = SaveTensorWithFixedBuffer(
      tensor_name + "-versions", writer, dump_buffer.get(),
      bytes_limit, &version_dump_iter,
      TensorShape({version_vec_.size()}));
  if (!s.ok())
    return s;

  EVVectorDataDumpIterator<int64> freq_dump_iter(freq_vec_);
  s = SaveTensorWithFixedBuffer(
      tensor_name + "-freqs", writer, dump_buffer.get(),
      bytes_limit, &freq_dump_iter,
      TensorShape({freq_vec_.size()}));
  if (!s.ok())
    return s;

  EVVectorDataDumpIterator<K> filtered_key_dump_iter(key_filter_vec_);
  s = SaveTensorWithFixedBuffer(
      tensor_name + "-keys_filtered", writer, dump_buffer.get(),
      bytes_limit, &filtered_key_dump_iter,
      TensorShape({key_filter_vec_.size()}));
  if (!s.ok())
    return s;

  EVVectorDataDumpIterator<int64>
      filtered_version_dump_iter(version_filter_vec_);
  s = SaveTensorWithFixedBuffer(
      tensor_name + "-versions_filtered",
      writer, dump_buffer.get(),
      bytes_limit, &filtered_version_dump_iter,
      TensorShape({version_filter_vec_.size()}));
  if (!s.ok())
    return s;

  EVVectorDataDumpIterator<int64>
      filtered_freq_dump_iter(freq_filter_vec_);
  s = SaveTensorWithFixedBuffer(
      tensor_name + "-freqs_filtered",
      writer, dump_buffer.get(),
      bytes_limit, &filtered_freq_dump_iter,
      TensorShape({freq_filter_vec_.size()}));
  if (!s.ok())
    return s;

  EVVectorDataDumpIterator<int32>
      part_offset_dump_iter(part_offset_);
  s = SaveTensorWithFixedBuffer(
      tensor_name + "-partition_offset",
      writer, dump_buffer.get(),
      bytes_limit, &part_offset_dump_iter,
      TensorShape({part_offset_.size()}));
  if (!s.ok())
    return s;

  EVVectorDataDumpIterator<int32>
      part_filter_offset_dump_iter(part_filter_offset_);
  s = SaveTensorWithFixedBuffer(
      tensor_name + "-partition_filter_offset",
      writer, dump_buffer.get(),
      bytes_limit, &part_filter_offset_dump_iter,
      TensorShape({part_filter_offset_.size()}));
  if (!s.ok())
    return s;

  return Status::OK();
}

#define REGISTER_KERNELS(ktype, vtype)                               \
  template Status EmbeddingVarCkptData<ktype, vtype>::ExportToCkpt(  \
      const string&, BundleWriter*, int64, ValueIterator<vtype>*); 
#define REGISTER_KERNELS_ALL_INDEX(type)                             \
  REGISTER_KERNELS(int32, type)                                      \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS
}// namespace embedding
}// namespace tensorflow
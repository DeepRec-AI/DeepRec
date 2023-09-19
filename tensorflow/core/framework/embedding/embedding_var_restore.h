/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_RESTORE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_RESTORE_H_

#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/filter_policy.h"
#include "tensorflow/core/framework/embedding/storage.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

template<typename K, typename V>
class EmbeddingVar;

namespace {
  const size_t kBufferSize = 8 << 20;
  constexpr char kPartStr[] = "part_";
  
  constexpr char kPartOffsetTensorSuffsix[] = "-partition_offset";
  constexpr char kPartFilterOffsetTensorSuffsix[] =
      "-partition_filter_offset";
  constexpr char kKeySuffix[] = "-keys";
  constexpr char kValueSuffix[] = "-values";
  constexpr char kVersionSuffix[] = "-versions";
  constexpr char kFreqSuffix[] = "-freqs";

  constexpr char kIncrPartOffsetTensorSuffsix[] = "-incr_partition_offset";
  constexpr char kIncrKeySuffix[] = "-sparse_incr_keys";
  constexpr char kIncrValueSuffix[] = "-sparse_incr_values";
  constexpr char kIncrVersionSuffix[] = "-sparse_incr_versions";
  constexpr char kIncrFreqSuffix[] = "-sparse_incr_freqs";
}  // namespace

template <typename K>
int64 ReadRecord(BundleReader* reader, const string& record_key, K** buffer);

template <typename K>
struct RestoreSSDBuffer {
  int64* file_list_buf = nullptr;
  int64* invalid_record_count_list_buf = nullptr;
  int64* record_count_list_buf = nullptr;
  K* key_list_buf = nullptr;
  int64* key_file_id_list_buf = nullptr;
  int64* key_offset_list_buf = nullptr;
  int64 num_of_keys = 0;
  int64 num_of_files = 0;

  explicit RestoreSSDBuffer(BundleReader* ssd_record_reader) {
    num_of_files = ReadRecord(ssd_record_reader, "files", &file_list_buf);

    ReadRecord(ssd_record_reader, "invalid_record_count",
               &invalid_record_count_list_buf);
    ReadRecord(ssd_record_reader, "record_count", &record_count_list_buf);
    num_of_keys = ReadRecord(ssd_record_reader, "keys", &key_list_buf);

    ReadRecord(ssd_record_reader, "keys_file_id", &key_file_id_list_buf);
    ReadRecord(ssd_record_reader, "keys_offset", &key_offset_list_buf);
  }

  ~RestoreSSDBuffer() {
    delete[] file_list_buf;
    delete[] invalid_record_count_list_buf;
    delete[] record_count_list_buf;
    delete[] key_list_buf;
    delete[] key_file_id_list_buf;
    delete[] key_offset_list_buf;
  }
};

struct RestoreArgs {
  std::string m_file_name_string;
  std::string m_name_string;
  std::string m_tensor_key;
  std::string m_tensor_value;
  std::string m_tensor_version;
  std::string m_tensor_freq;
  std::vector<int> m_loaded_parts;
  int64 m_partition_id;
  int64 m_partition_num;
  int64 m_idx;
  int m_old_dim;
  bool m_is_incr;
  bool m_reset_version;
  bool m_has_freq;
  bool m_has_filter;
  bool m_is_oldform;
  RestoreArgs(const std::string name_string,
              const std::string file_name_string,
              int64 partition_id,
              int64 partition_num,
              bool is_incr,
              bool reset_version):
      m_name_string(name_string), m_file_name_string(file_name_string),
      m_partition_id(partition_id), m_partition_num(partition_num),
      m_idx(0), m_old_dim(0), m_is_incr(is_incr),
      m_reset_version(reset_version), m_has_freq(true),
      m_has_filter(true), m_is_oldform(false) {}
  RestoreArgs() = default;
};

template <typename K, typename V>
class CheckpointLoader {
 public:
  CheckpointLoader(embedding::Storage<K, V>* storage, EmbeddingVar<K, V>* ev,
                   FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
                   const std::string& name_string,
                   const std::string& file_name_string, int64 partition_id,
                   int64 partition_num, bool is_incr, bool reset_version,
                   BundleReader* reader)
      : storage_(storage), ev_(ev), filter_(filter), reader_(reader) {
    restore_args_ = RestoreArgs(name_string, file_name_string, partition_id,
                                partition_num, is_incr, reset_version);
  }

  void RestoreCkpt(const EmbeddingConfig& emb_config,
                   const Eigen::GpuDevice* device) {
    /* Step 1: Restore SSD ckpt Data (Optional)
       Step 2; Restore model ckpt */
    RestoreSSD();

    std::vector<std::string> tensor_name_vec;
    InitPartNumAndLoadedParts(tensor_name_vec);

    RestoreBuffer restore_buff(kBufferSize);
    for (auto& tensor_name : tensor_name_vec) {
      RestoreInternal(tensor_name, emb_config, device, restore_buff);
    }

  }

  void RestoreInternal(const std::string& name_string,
                       const EmbeddingConfig& emb_config,
                       const Eigen::GpuDevice* device,
                       RestoreBuffer& restore_buff);

 private:
  void RestoreSSD();

  bool IsOldCheckpoint(const std::string& curr_partid_str,
                       const std::string& kPartOffsetTensorSuffsix);

  void InitPartNumAndLoadedParts(std::vector<std::string>& tensor_name_vec);

  Status EVInitTensorNameAndShape(const std::string& tensor_name);

  Status EVRestoreFeatures(int tot_key_num, int64 key_part_offset,
                           int64 value_part_offset, int64 version_part_offset,
                           int64 freq_part_offset, RestoreBuffer& restore_buff,
                           int64 new_dim, const EmbeddingConfig& emb_config,
                           const Eigen::GpuDevice* device);

  Status EVRestoreFilteredFeatures(
      int64 subpart_id, int64 value_len, RestoreBuffer& restore_buff,
      typename TTypes<int32>::Flat part_filter_offset_flat,
      const EmbeddingConfig& emb_config, const Eigen::GpuDevice* device);

  Status RestoreCustomDim(int new_dim, int read_key_num,
                          size_t value_unit_bytes, size_t value_bytes_read,
                          size_t value_unit_bytes_new,
                          RestoreBuffer& restore_buff) {
    bool restore_customDim;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_EV_RESTORE_CUSTOM_DIM", false,
                                   &restore_customDim));
    if (restore_customDim && ev_->IsUseHbm()) {
      return errors::FailedPrecondition(
          "HBM EV not and custom dim,"
          "are not supported used together");
    }
    if (restore_customDim && restore_args_.m_old_dim != new_dim) {
      VLOG(2) << "restore, read_value_reshape dim: from "
              << restore_args_.m_old_dim << " to " << new_dim;
      if (read_key_num * value_unit_bytes != value_bytes_read) {
        return tensorflow::errors::FailedPrecondition(
            "Expected read_key_num * value_unit_bytes == "
            "value_bytes_read, but got read_key_num * value_unit_bytes "
            "!= value_bytes_read!");
      }

      std::unique_ptr<char[]> tmp_ptr(new char[kBufferSize]);
      size_t read_once = std::min(value_unit_bytes, value_unit_bytes_new);
      for (int i = 0; i < read_key_num; ++i) {
        memcpy(tmp_ptr.get() + i * value_unit_bytes_new,
               restore_buff.value_buffer + i * value_unit_bytes, read_once);
        if (restore_args_.m_old_dim >= new_dim) continue;
        auto p = ev_->GetDefaultValue(restore_args_.m_idx++);
        memcpy(tmp_ptr.get() + i * value_unit_bytes_new + value_unit_bytes,
               p + value_unit_bytes, value_unit_bytes_new - value_unit_bytes);
      }
      auto tmp = tmp_ptr.release();
      tmp_ptr.reset(restore_buff.value_buffer);
      restore_buff.value_buffer = tmp;
    }
    return Status::OK();
  }

 private:
  embedding::Storage<K, V>* storage_;
  EmbeddingVar<K, V>* ev_;
  FilterPolicy<K, V, EmbeddingVar<K, V>>* filter_;
  BundleReader* reader_;
  RestoreArgs restore_args_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_RESTORE_H_

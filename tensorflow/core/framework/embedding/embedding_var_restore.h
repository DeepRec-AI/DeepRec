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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/filter_policy.h"
#include "tensorflow/core/framework/embedding/storage.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

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
int64 ReadRecord(BundleReader* reader, const string& record_key, K** buffer) {
  TensorShape shape;
  Status st;
  st = reader->LookupTensorShape(record_key, &shape);
  if (!st.ok()) {
    LOG(FATAL) << "Restore record " << record_key << " failed";
  }
  st = reader->LookupHeader(record_key, sizeof(K) * shape.dim_size(0));
  if (!st.ok()) {
    LOG(FATAL) << "Restore record " << record_key << " failed";
  }
  size_t bytes_read = 0;
  *buffer = new K[shape.dim_size(0)];
  st = reader->LookupSegment(record_key, sizeof(K) * shape.dim_size(0),
                             (char*)*buffer, bytes_read);
  if (!st.ok()) {
    LOG(FATAL) << "Restore record " << record_key << " failed";
  }
  return shape.dim_size(0);
}

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
                       RestoreBuffer& restore_buff) {
    Status s = EVInitTensorNameAndShape(name_string);
    if (!s.ok()) {
      LOG(ERROR) << "EVInitTensorNameAndShape fail:" << s.ToString();
      return;
    }

    Tensor part_offset_tensor;
    Tensor part_filter_offset_tensor;
    if (!restore_args_.m_is_oldform) {
      /****** InitPartOffsetTensor ******/
      TensorShape part_offset_shape, part_filter_offset_shape;
      DataType part_offset_type, part_filter_offset_type;
      string offset_tensor_name;
      if (!restore_args_.m_is_incr) {
        offset_tensor_name = name_string + kPartOffsetTensorSuffsix;
      } else {
        offset_tensor_name = name_string + kIncrPartOffsetTensorSuffsix;
      }
      
      string offset_filter_tensor_name =
          name_string + kPartFilterOffsetTensorSuffsix;
      Status s = reader_->LookupDtypeAndShape(
          offset_tensor_name, &part_offset_type, &part_offset_shape);
      if (!s.ok()) {
        LOG(ERROR) << "EV restoring fail:" << s.error_message();
      }
      s = reader_->LookupDtypeAndShape(offset_filter_tensor_name,
                                       &part_filter_offset_type,
                                       &part_filter_offset_shape);
      if (!s.ok()) {
        LOG(ERROR) << "EV restoring fail: " << s.error_message();
      }
      part_offset_tensor =
          Tensor(cpu_allocator(), part_offset_type, part_offset_shape);
      part_filter_offset_tensor = Tensor(
          cpu_allocator(), part_filter_offset_type, part_filter_offset_shape);
      s = reader_->Lookup(offset_tensor_name, &part_offset_tensor);
      if (!s.ok()) {
        LOG(ERROR) << "EV restoring fail:" << s.error_message();
      }

      s = reader_->Lookup(offset_filter_tensor_name,
                          &part_filter_offset_tensor);
      if (!s.ok()) {
        LOG(ERROR) << "EV restoring fail: " << s.error_message();
      }
    }
    auto part_offset_flat = part_offset_tensor.flat<int32>();
    auto part_filter_offset_flat = part_filter_offset_tensor.flat<int32>();
    
    if (restore_args_.m_is_oldform) {
      VLOG(1) << "old form, EV name:" << name_string
              << ", partition_id:" << restore_args_.m_partition_id
              << ", new partition num:" << restore_args_.m_partition_num;
      int64 new_dim = ev_->ValueLen();
      TensorShape key_shape;
      Status st =
          reader_->LookupTensorShape(restore_args_.m_tensor_key, &key_shape);
      if (!st.ok()) {
      }
      int tot_key_num = key_shape.dim_size(0);
      Status s = EVRestoreFeatures(tot_key_num, 0, 0, 0, 0, restore_buff,
                                   new_dim, emb_config, device);
      if (!s.ok()) {
        LOG(ERROR) << "EVRestoreFeaturesOld fail: " << s.error_message();
      }
    } else {
      int64 new_dim = ev_->ValueLen();
      VLOG(1) << "new form checkpoint... :" << name_string
              << " , partition_id:" << restore_args_.m_partition_id
              << " , partition_num:" << restore_args_.m_partition_num;
      for (size_t i = 0; i < restore_args_.m_loaded_parts.size(); i++) {
        int subpart_id = restore_args_.m_loaded_parts[i];
        size_t value_unit_bytes = sizeof(V) * restore_args_.m_old_dim;
        size_t value_unit_bytes_new = sizeof(V) * new_dim;
        int subpart_offset = part_offset_flat(subpart_id);
        int tot_key_num = part_offset_flat(subpart_id + 1) - subpart_offset;
        int64 key_part_offset = subpart_offset * sizeof(K);
        int64 value_part_offset =
            subpart_offset * sizeof(V) * restore_args_.m_old_dim;
        int64 version_part_offset = subpart_offset * sizeof(int64);
        int64 freq_part_offset = subpart_offset * sizeof(int64);
        VLOG(1) << "dynamically load ev : " << name_string
                << ", subpartid:" << subpart_id;

        EVRestoreFeatures(tot_key_num, key_part_offset, value_part_offset,
                          version_part_offset, freq_part_offset, restore_buff,
                          new_dim, emb_config, device);

        if (restore_args_.m_has_filter) {
          Status s = EVRestoreFilteredFeatures(
              subpart_id, new_dim, restore_buff, part_filter_offset_flat,
              emb_config, device);
          if (!s.ok()) {
            LOG(ERROR) << "EVRestoreFilteredFeatures fail: " << s.error_message();
          }
        }
      }
    }
  }

 private:
  void RestoreSSD() {
    std::string name_string_temp(restore_args_.m_name_string);
    std::string new_str = "_";
    int64 pos = name_string_temp.find("/");
    while (pos != std::string::npos) {
      name_string_temp.replace(pos, 1, new_str.data(), 1);
      pos = name_string_temp.find("/");
    }
    std::string ssd_record_file_name = restore_args_.m_file_name_string + "-" +
                                       name_string_temp + "-ssd_record";
    if (Env::Default()->FileExists(ssd_record_file_name + ".index").ok()) {
      std::string ssd_emb_file_name = restore_args_.m_file_name_string + "-" +
                                      name_string_temp + "-emb_files";
      BundleReader ssd_record_reader(Env::Default(), ssd_record_file_name);
      RestoreSSDBuffer<K> ssd_buffer(&ssd_record_reader);
      VLOG(1) << "Loading SSD record... " << ssd_record_file_name;
      storage_->RestoreSSD(ev_->GetEmbeddingIndex(),
                           ev_->GetEmbeddingSlotNum(), ev_->ValueLen(),
                           ssd_emb_file_name, ev_, ssd_buffer);
    }
  }

  bool IsOldCheckpoint(const std::string& curr_partid_str,
                       const std::string& kPartOffsetTensorSuffsix) {
    if (restore_args_.m_name_string.find(kPartStr) == std::string::npos) {
      string tensor_name = restore_args_.m_name_string;
      TensorShape part_offset_shape;
      DataType part_offset_type;
      Status st =
          reader_->LookupDtypeAndShape(tensor_name + kPartOffsetTensorSuffsix,
                                       &part_offset_type, &part_offset_shape);
      if (st.ok()) return false;

      string part_id = std::to_string(0);
      tensor_name = restore_args_.m_name_string + "/" + kPartStr + part_id;

      Status form_st =
          reader_->LookupDtypeAndShape(tensor_name + kPartOffsetTensorSuffsix,
                                       &part_offset_type, &part_offset_shape);
      if (form_st.ok()) return false;
    } else {
      string part_id = std::to_string(0);
      size_t part_pos = restore_args_.m_name_string.find(kPartStr);
      size_t part_size = strlen(kPartStr);
      size_t cur_part_size = curr_partid_str.size();

      string pre_subname = restore_args_.m_name_string.substr(0, part_pos);
      string post_subname = restore_args_.m_name_string.substr(
          part_pos + part_size + cur_part_size);
      string tensor_name = pre_subname + kPartStr + part_id + post_subname;

      TensorShape part_offset_shape;
      DataType part_offset_type;
      Status form_st =
          reader_->LookupDtypeAndShape(tensor_name + kPartOffsetTensorSuffsix,
                                       &part_offset_type, &part_offset_shape);
      if (form_st.ok()) return false;
      pre_subname =
          restore_args_.m_name_string.substr(0, part_pos - 1); /* var1*/
      post_subname = restore_args_.m_name_string.substr(part_pos + part_size +
                                                        cur_part_size);
      tensor_name = pre_subname + post_subname;

      Status st =
          reader_->LookupDtypeAndShape(tensor_name + kPartOffsetTensorSuffsix,
                                       &part_offset_type, &part_offset_shape);
      if (st.ok()) return false;
    }

    return true;
  }

  void InitPartNumAndLoadedParts(std::vector<std::string>& tensor_name_vec) {
    std::string tmp_key_suffix;
    std::string tmp_kPartOffsetTensorSuffsix;
    if (!restore_args_.m_is_incr) {
      tmp_key_suffix = kKeySuffix;
      tmp_kPartOffsetTensorSuffsix = kPartOffsetTensorSuffsix;
    } else {
      tmp_key_suffix = kIncrKeySuffix;
      tmp_kPartOffsetTensorSuffsix = kIncrPartOffsetTensorSuffsix;
    }

    restore_args_.m_loaded_parts.reserve(kSavedPartitionNum);
    int orig_partnum = 0;
    const string& curr_partid_str = std::to_string(restore_args_.m_partition_id);
    size_t part_pos = restore_args_.m_name_string.find(kPartStr);

    if (IsOldCheckpoint(curr_partid_str, tmp_kPartOffsetTensorSuffsix)) {
      restore_args_.m_is_oldform = true;
    }

    if (part_pos == std::string::npos) {
      for (;; orig_partnum++) {
        string part_id = std::to_string(orig_partnum);
        string tensor_name =
            restore_args_.m_name_string + "/" + kPartStr + part_id;
        string tensor_key = tensor_name + tmp_key_suffix;
        TensorShape key_shape;
        Status st = reader_->LookupTensorShape(tensor_key, &key_shape);
        if (!st.ok()) {
          break;
        }
        tensor_name_vec.emplace_back(tensor_name);
      }
      if (orig_partnum == 0) {
        tensor_name_vec.emplace_back(restore_args_.m_name_string);
      }
      for (int i = 0; i < kSavedPartitionNum; ++i) {
        restore_args_.m_loaded_parts.push_back(i);
      }
    } else {
      for (;; orig_partnum++) {
        string part_id = std::to_string(orig_partnum);
        string pre_subname = restore_args_.m_name_string.substr(0, part_pos);
        string post_subname = restore_args_.m_name_string.substr(
            part_pos + strlen(kPartStr) + curr_partid_str.size());
        string tensor_name = pre_subname + kPartStr + part_id + post_subname;
        string tensor_key = tensor_name + tmp_key_suffix;
        TensorShape key_shape;
        Status st = reader_->LookupTensorShape(tensor_key, &key_shape);
        if (!st.ok()) {
          break;
        }
        tensor_name_vec.emplace_back(tensor_name);
      }
      if (orig_partnum == 0) {
        string pre_subname = restore_args_.m_name_string.substr(0, part_pos - 1);
        string post_subname = restore_args_.m_name_string.substr(
            part_pos + strlen(kPartStr) + curr_partid_str.size());
        string tmp_name = pre_subname + post_subname;
        tensor_name_vec.emplace_back(tmp_name);
      }
      for (int i = 0; i < kSavedPartitionNum; i++) {
        if (i % restore_args_.m_partition_num == restore_args_.m_partition_id) {
          restore_args_.m_loaded_parts.push_back(i);
        }
      }
    }
    for (auto& tensor_name : tensor_name_vec) {
      VLOG(1) << "**** " << restore_args_.m_name_string << " " << tensor_name
              << " ****";
    }
  }

  Status EVInitTensorNameAndShape(const std::string& tensor_name) {
    if (!restore_args_.m_is_incr) {
      restore_args_.m_tensor_key = tensor_name + kKeySuffix;
      restore_args_.m_tensor_value = tensor_name + kValueSuffix;
      restore_args_.m_tensor_version = tensor_name + kVersionSuffix;
      restore_args_.m_tensor_freq = tensor_name + kFreqSuffix;
    } else {
      restore_args_.m_tensor_key = tensor_name + kIncrKeySuffix;
      restore_args_.m_tensor_value = tensor_name + kIncrValueSuffix;
      restore_args_.m_tensor_version = tensor_name + kIncrVersionSuffix;
      restore_args_.m_tensor_freq = tensor_name + kIncrFreqSuffix;
    }

    TensorShape key_shape, value_shape, version_shape, freq_shape;

    Status st =
        reader_->LookupTensorShape(restore_args_.m_tensor_key, &key_shape);
    if (!st.ok()) {
      return st;
    }
    st = reader_->LookupTensorShape(restore_args_.m_tensor_value, &value_shape);
    if (!st.ok()) {
      return st;
    }
    st = reader_->LookupTensorShape(restore_args_.m_tensor_version,
                                    &version_shape);
    if (!st.ok()) {
      return st;
    }
    st = reader_->LookupHeader(restore_args_.m_tensor_key,
                               sizeof(K) * key_shape.dim_size(0));
    if (!st.ok()) {
      return st;
    }
    st = reader_->LookupHeader(restore_args_.m_tensor_value,
                               sizeof(V) * value_shape.dim_size(0) *
                               value_shape.dim_size(1));
    if (!st.ok()) {
      return st;
    }
    st = reader_->LookupHeader(restore_args_.m_tensor_version,
                               sizeof(int64) * version_shape.dim_size(0));
    if (!st.ok()) {
      return st;
    }
    st = reader_->LookupTensorShape(restore_args_.m_tensor_freq, &freq_shape);
    if (!st.ok()) {
      if (st.code() == error::NOT_FOUND) {
        freq_shape = version_shape;
      } else {
        return st;
      }
    }
    st = reader_->LookupHeader(restore_args_.m_tensor_freq,
                               sizeof(int64) * freq_shape.dim_size(0));
    if (!st.ok()) {
      if (st.code() == error::NOT_FOUND) {
        restore_args_.m_has_freq = false;
      } else {
        return st;
      }
    }
    restore_args_.m_old_dim = value_shape.dim_size(1);

    if (!restore_args_.m_is_oldform) {
      TensorShape key_filter_shape, version_filter_shape, freq_filter_shape;
      st = reader_->LookupTensorShape(restore_args_.m_tensor_key + "_filtered",
                                      &key_filter_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          key_filter_shape = key_shape;
          restore_args_.m_has_filter = false;
        } else {
          return st;
        }
      }
      st = reader_->LookupTensorShape(
          restore_args_.m_tensor_version + "_filtered", &version_filter_shape);
      if ((!st.ok()) && (st.code() != error::NOT_FOUND)) {
        return st;
      }
      st = reader_->LookupHeader(restore_args_.m_tensor_key + "_filtered",
                                 sizeof(K) * key_filter_shape.dim_size(0));
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          restore_args_.m_has_filter = false;
        } else {
          return st;
        }
      }
      st = reader_->LookupHeader(restore_args_.m_tensor_version + "_filtered",
                                 sizeof(K) * version_filter_shape.dim_size(0));
      if (!st.ok()) {
        return st;
      }
      st = reader_->LookupTensorShape(restore_args_.m_tensor_freq + "_filtered",
                                      &freq_filter_shape);
      if (!st.ok()) {
        if (st.code() == error::NOT_FOUND) {
          freq_filter_shape = freq_shape;
        } else {
          return st;
        }
      }

      st = reader_->LookupHeader(restore_args_.m_tensor_freq + "_filtered",
                                 sizeof(K) * freq_filter_shape.dim_size(0));
      if (!st.ok() && st.code() != error::NOT_FOUND) {
        return st;
      }
    }
    return st;
  }

  Status EVRestoreFeatures(int tot_key_num, int64 key_part_offset,
                           int64 value_part_offset, int64 version_part_offset,
                           int64 freq_part_offset, RestoreBuffer& restore_buff,
                           int64 new_dim, const EmbeddingConfig& emb_config,
                           const Eigen::GpuDevice* device) {
    size_t value_unit_bytes = sizeof(V) * restore_args_.m_old_dim;
    size_t value_unit_bytes_new = sizeof(V) * new_dim;
    int64 tot_key_bytes_read(0);
    int64 tot_value_bytes_read(0);
    int64 tot_version_bytes_read(0);
    int64 tot_freq_bytes_read(0);
    size_t key_bytes_read = 0;
    size_t value_bytes_read = 0;
    size_t version_bytes_read = 0;
    size_t freq_bytes_read = 0;

    while (tot_key_num > 0) {
      size_t read_key_num = std::min(
          std::min(kBufferSize / sizeof(K), kBufferSize / value_unit_bytes),
                   kBufferSize / sizeof(int64));
      read_key_num = std::min(read_key_num, kBufferSize / value_unit_bytes_new);
      read_key_num = std::min((int)read_key_num, tot_key_num);
      reader_->LookupSegmentOffset(
          restore_args_.m_tensor_key, key_part_offset + tot_key_bytes_read,
          read_key_num * sizeof(K), restore_buff.key_buffer, key_bytes_read);
      reader_->LookupSegmentOffset(
          restore_args_.m_tensor_value, value_part_offset + tot_value_bytes_read,
          read_key_num * value_unit_bytes, restore_buff.value_buffer,
          value_bytes_read);
      if (!restore_args_.m_reset_version) {
        reader_->LookupSegmentOffset(
            restore_args_.m_tensor_version,
            version_part_offset + tot_version_bytes_read,
            read_key_num * sizeof(int64), restore_buff.version_buffer,
            version_bytes_read);
        if (version_bytes_read == 0) {
          memset(restore_buff.version_buffer, -1, sizeof(int64) * read_key_num);
        }
      } else {
        int64* version_tmp = (int64*)restore_buff.version_buffer;
        memset(version_tmp, 0, read_key_num * sizeof(int64));
      }

      if (restore_args_.m_has_freq) {
        reader_->LookupSegmentOffset(
            restore_args_.m_tensor_freq, freq_part_offset + tot_freq_bytes_read,
            read_key_num * sizeof(int64), restore_buff.freq_buffer,
            freq_bytes_read);
        if (freq_bytes_read == 0) {
          int64* freq_tmp = (int64*)restore_buff.freq_buffer;
          for (int64 i = 0; i < read_key_num; i++) {
            freq_tmp[i] = (ev_->MinFreq() == 0) ? 1 : ev_->MinFreq();
          }
        }
      } else {
        int64* freq_tmp = (int64*)restore_buff.freq_buffer;
        for (int64 i = 0; i < read_key_num; i++) {
          freq_tmp[i] = (ev_->MinFreq() == 0) ? 1 : ev_->MinFreq();
        }
      }
      if (key_bytes_read > 0) {
        read_key_num = key_bytes_read / sizeof(K);
        Status st = RestoreCustomDim(new_dim, read_key_num, value_unit_bytes,
                                     value_bytes_read, value_unit_bytes_new,
                                     restore_buff);
        if (!st.ok()) {
          LOG(FATAL) << "EV Restore fail:" << st.ToString();
        }

        st = storage_->RestoreFeatures(
            read_key_num, kSavedPartitionNum, restore_args_.m_partition_id,
            restore_args_.m_partition_num, new_dim, false, restore_args_.m_is_incr,
            emb_config, device,
            filter_, restore_buff);
        if (!st.ok()) {
          LOG(FATAL) << "EV Restore fail:" << st.ToString();
        }
      }

      tot_key_num -= read_key_num;
      tot_key_bytes_read += key_bytes_read;
      tot_value_bytes_read += value_bytes_read;
      tot_version_bytes_read += version_bytes_read;
      tot_freq_bytes_read += freq_bytes_read;
    }

    return Status::OK();
  }

  Status EVRestoreFilteredFeatures(
      int64 subpart_id, int64 value_len, RestoreBuffer& restore_buff,
      typename TTypes<int32>::Flat part_filter_offset_flat,
      const EmbeddingConfig& emb_config, const Eigen::GpuDevice* device) {
    int subpart_filter_offset = part_filter_offset_flat(subpart_id);
    int tot_key_filter_num =
        part_filter_offset_flat(subpart_id + 1) - subpart_filter_offset;
    int64 key_filter_part_offset = subpart_filter_offset * sizeof(K);
    int64 version_filter_part_offset = subpart_filter_offset * sizeof(int64);
    int64 freq_filter_part_offset = subpart_filter_offset * sizeof(int64);

    VLOG(1) << "key_filter_num: " << tot_key_filter_num
            << ", subpart_filter_offset: " << subpart_filter_offset;

    size_t key_filter_bytes_read = 0;
    size_t version_filter_bytes_read = 0;
    size_t freq_filter_bytes_read = 0;

    while (tot_key_filter_num > 0) {
      size_t read_key_num =
          std::min(kBufferSize / sizeof(K), kBufferSize / sizeof(int64));
      read_key_num = std::min((int)read_key_num, tot_key_filter_num);
      reader_->LookupSegmentOffset(
          restore_args_.m_tensor_key + "_filtered",
          key_filter_part_offset + key_filter_bytes_read,
          read_key_num * sizeof(K), restore_buff.key_buffer,
          key_filter_bytes_read);
      if (!restore_args_.m_reset_version) {
        reader_->LookupSegmentOffset(
            restore_args_.m_tensor_version + "_filtered",
            version_filter_part_offset + version_filter_bytes_read,
            read_key_num * sizeof(int64), restore_buff.version_buffer,
            version_filter_bytes_read);
      } else {
        int64* version_tmp = (int64*)restore_buff.version_buffer;
        memset(version_tmp, 0, read_key_num * sizeof(int64));
      }
      reader_->LookupSegmentOffset(
          restore_args_.m_tensor_freq + "_filtered",
          freq_filter_part_offset + freq_filter_bytes_read,
          read_key_num * sizeof(int64), restore_buff.freq_buffer,
          freq_filter_bytes_read);
      if (key_filter_bytes_read > 0) {
        read_key_num = key_filter_bytes_read / sizeof(K);
        VLOG(2) << "restore, read_key_num:" << read_key_num;
        Status st = storage_->RestoreFeatures(
            read_key_num, kSavedPartitionNum, restore_args_.m_partition_id,
            restore_args_.m_partition_num, value_len, true, restore_args_.m_is_incr,
            emb_config, device,
            filter_, restore_buff);
        if (!st.ok()) return st;
        tot_key_filter_num -= read_key_num;
      }
    }
    return Status::OK();
  }

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

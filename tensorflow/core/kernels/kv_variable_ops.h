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

#ifndef TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_
#define TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

template<class T>
class EVKeyDumpIterator: public  DumpIterator<T> {
 public:
  EVKeyDumpIterator(std::vector<T>& key_list):key_list_(key_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < key_list_.size();
  }

  T Next() {
    return key_list_[keys_idx_++];
  }

 private:
  int64 keys_idx_;
  std::vector<T>& key_list_;
};

template<class K, class T>
class EVValueDumpIterator: public  DumpIterator<T> {
 public:
  EVValueDumpIterator(EmbeddingVar<K, T>*& ev,
      std::vector<T* >& valueptr_list)
        : ev_(ev),
          valueptr_list_(valueptr_list) {
    keys_idx_ = 0;
    col_idx_ = 0;
  }

  bool HasNext() const {
    if (keys_idx_ < valueptr_list_.size()) {
      if (keys_idx_ < valueptr_list_.size() - 1)
        return true;
      else
        return col_idx_ < ev_->ValueLen();
    } else
      return false;
  }

  T Next() {
    if (col_idx_ >= ev_->ValueLen()) {
      keys_idx_++;
      col_idx_ = 0;
    }
    Eigen::array<Eigen::DenseIndex, 1> dims({ev_->ValueLen()});
    typename TTypes<T>::Flat value_flat =
      typename TTypes<T>::Flat(valueptr_list_[keys_idx_], dims);
    return value_flat(col_idx_++);
  }

 private:
  EmbeddingVar<K, T>* ev_;
  std::vector<T* >& valueptr_list_;
  int64 keys_idx_;
  int64 col_idx_;
};

template<class T>
class EVVersionDumpIterator: public  DumpIterator<T> {
 public:
  EVVersionDumpIterator(std::vector<T >& version_list)
      : version_list_(version_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < version_list_.size();
  }

  T Next() {
    return version_list_[keys_idx_++];
  }

 private:
  std::vector<T>& version_list_;
  int64 keys_idx_;
};

template<class T>
class EVFreqDumpIterator: public  DumpIterator<T> {
 public:
  EVFreqDumpIterator(std::vector<T>& freq_list) : freq_list_(freq_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < freq_list_.size();
  }

  T Next() {
    return freq_list_[keys_idx_++];
  }

 private:
  std::vector<T>& freq_list_;
  int64 keys_idx_;
};

template<class T>
class EVOffsetDumpIterator: public  DumpIterator<T> {
 public:
  EVOffsetDumpIterator(std::vector<T>& offset_list)
      : offset_list_(offset_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < offset_list_.size();
  }

  T Next() {
    return offset_list_[keys_idx_++];
  }

 private:
  std::vector<T>& offset_list_;
  int64 keys_idx_;
};

template <class K, class V>
Status GetInputEmbeddingVar(OpKernelContext* ctx, int input,
                            EmbeddingVar<K, V>** var) {
  if (LookupResource(ctx, HandleFromInput(ctx, input), var).ok()) {
    return Status::OK();
  } else {
    return errors::Internal("Invalid versioned variable reference.");
  }
}

template <class K>
void DumpSsdIndexMeta(
    SsdRecordDescriptor<K>& ssd_rec_desc,
    const std::string& prefix,
    const std::string& var_name) {
  std::fstream fs;
  std::string var_name_temp(var_name);
  std::string new_str = "_";
  int64 pos = var_name_temp.find("/");
  while (pos != std::string::npos) {
    var_name_temp.replace(pos, 1, new_str.data(), 1);
    pos =var_name_temp.find("/");
  }

  std::string ssd_record_path =
      prefix + "-" + var_name_temp + "-ssd_record";

  BundleWriter ssd_record_writer(Env::Default(),
                                 ssd_record_path);
  typedef EVFreqDumpIterator<int64> Int64DataDumpIterator;
  size_t bytes_limit = 8 << 20;
  char* dump_buffer = new char[bytes_limit];

  int64 num_of_keys = ssd_rec_desc.key_list.size();
  EVKeyDumpIterator<K> keys_iter(ssd_rec_desc.key_list);
  SaveTensorWithFixedBuffer(
      "keys",
      &ssd_record_writer, dump_buffer,
      bytes_limit, &keys_iter,
      TensorShape({num_of_keys}));

  Int64DataDumpIterator key_file_id_iter(ssd_rec_desc.key_file_id_list);
  SaveTensorWithFixedBuffer(
      "keys_file_id",
      &ssd_record_writer, dump_buffer,
      bytes_limit, &key_file_id_iter,
      TensorShape({num_of_keys}));

  Int64DataDumpIterator key_offset_iter(ssd_rec_desc.key_offset_list);
  SaveTensorWithFixedBuffer(
      "keys_offset",
      &ssd_record_writer, dump_buffer,
      bytes_limit, &key_offset_iter,
      TensorShape({num_of_keys}));

  int64 num_of_files = ssd_rec_desc.file_list.size();
  Int64DataDumpIterator files_iter(ssd_rec_desc.file_list);
  SaveTensorWithFixedBuffer(
      "files",
      &ssd_record_writer, dump_buffer,
      bytes_limit, &files_iter,
      TensorShape({num_of_files}));

  Int64DataDumpIterator
      invalid_record_count_iter(ssd_rec_desc.invalid_record_count_list);
  SaveTensorWithFixedBuffer(
      "invalid_record_count",
      &ssd_record_writer, dump_buffer,
      bytes_limit, &invalid_record_count_iter,
      TensorShape({num_of_files}));

  Int64DataDumpIterator
      record_count_iter(ssd_rec_desc.record_count_list);
  SaveTensorWithFixedBuffer(
      "record_count",
      &ssd_record_writer, dump_buffer,
      bytes_limit, &record_count_iter,
      TensorShape({num_of_files}));

  ssd_record_writer.Finish();
  delete[] dump_buffer;
}

template<class K>
void CopyEmbeddingfilesToCkptDir(
    const SsdRecordDescriptor<K>& ssd_rec_desc,
    const std::string& prefix,
    const std::string& var_name) {
  std::string var_name_temp(var_name);
  std::string new_str = "_";
  int64 pos = var_name_temp.find("/");
  while (pos != std::string::npos) {
    var_name_temp.replace(pos, 1, new_str.data(), 1);
    pos =var_name_temp.find("/");
  }

  std::string embedding_folder_path =
      prefix + "-" + var_name_temp + "-emb_files/";
  Status s = Env::Default()->CreateDir(embedding_folder_path);
  if (errors::IsAlreadyExists(s)) {
    int64 undeleted_files, undeleted_dirs;
    Env::Default()->
        DeleteRecursively(embedding_folder_path,
                          &undeleted_files,
                          &undeleted_dirs);
    Env::Default()->CreateDir(embedding_folder_path);
  }

  for (int64 i = 0; i < ssd_rec_desc.file_list.size(); i++) {
    int64 file_id = ssd_rec_desc.file_list[i];
    std::stringstream old_ss;
    old_ss << std::setw(4) << std::setfill('0') << file_id << ".emb";
    std::string file_path = ssd_rec_desc.file_prefix + old_ss.str();
    std::string file_name = file_path.substr(file_path.rfind("/"));
    std::stringstream new_ss;
    new_ss << file_id << ".emb";
    std::string new_file_path = embedding_folder_path + new_ss.str();
    Status s = Env::Default()->CopyFile(file_path, new_file_path);
    if (!s.ok()) {
      LOG(FATAL)<<"Copy file "<<file_path<<" failed!";
    }
  }
}

template <class K, class V>
Status DumpEmbeddingValues(EmbeddingVar<K, V>* ev,
    const string& tensor_key, BundleWriter* writer,
    Tensor* part_offset_tensor,
    const std::string& prefix = "") {
  std::vector<K> tot_key_list;
  std::vector<V*> tot_valueptr_list;
  std::vector<int64> tot_version_list;
  std::vector<int64> tot_freq_list;
  std::vector<K> tot_key_filter_list;
  std::vector<int64> tot_freq_filter_list;
  std::vector<int64> tot_version_filter_list;
  embedding::Iterator* it = nullptr;
  int64 num_of_keys = 0;
  //For the time being, only ev which uses SSD for storage,
  //ev->IsUsePersistentStorage() will get true.
  if (ev->IsUsePersistentStorage()) {
    SsdRecordDescriptor<K> ssd_rec_desc;
    num_of_keys =
        ev->GetSnapshotWithoutFetchPersistentEmb(
            &tot_key_list,
            &tot_valueptr_list,
            &tot_version_list,
            &tot_freq_list,
            &ssd_rec_desc);
    bool is_primary = (ev->GetEmbeddingIndex() == 0);
    if (is_primary) {
      DumpSsdIndexMeta(ssd_rec_desc, prefix, tensor_key);
      CopyEmbeddingfilesToCkptDir(ssd_rec_desc, prefix, tensor_key);
    }
  } else {
    num_of_keys = ev->GetSnapshot(
        &tot_key_list,
        &tot_valueptr_list,
        &tot_version_list,
        &tot_freq_list, &it);
  }

  VLOG(1) << "EV:" << tensor_key << ", save size:" << num_of_keys;
  int64 iterator_size = 0;
  int64 filter_iterator_size = 0;
  if (it != nullptr) {
    it->SwitchToAdmitFeatures();
    ev->storage()->iterator_mutex_lock();
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      ++iterator_size;
    }
    it->SwitchToFilteredFeatures();
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      ++filter_iterator_size;
    }
  }

  std::vector<std::vector<K> > key_list_parts;
  std::vector<std::vector<V* > > valueptr_list_parts;
  std::vector<std::vector<int64 > > version_list_parts;
  std::vector<std::vector<int64 > > freq_list_parts;

  std::vector<std::vector<K> > key_filter_list_parts;
  std::vector<std::vector<int64 > > version_filter_list_parts;
  std::vector<std::vector<int64 > > freq_filter_list_parts;

  std::vector<K> partitioned_tot_key_list;
  std::vector<V* > partitioned_tot_valueptr_list;
  std::vector<int64> partitioned_tot_version_list;
  std::vector<int64> partitioned_tot_freq_list;
  std::vector<K> partitioned_tot_key_filter_list;
  std::vector<int64> partitioned_tot_version_filter_list;
  std::vector<int64> partitioned_tot_freq_filter_list;
  std::vector<int64> part_filter_offset;

  key_list_parts.resize(kSavedPartitionNum);
  valueptr_list_parts.resize(kSavedPartitionNum);
  version_list_parts.resize(kSavedPartitionNum);
  freq_list_parts.resize(kSavedPartitionNum);
  key_filter_list_parts.resize(kSavedPartitionNum);
  version_filter_list_parts.resize(kSavedPartitionNum);
  freq_filter_list_parts.resize(kSavedPartitionNum);
  part_filter_offset.resize(kSavedPartitionNum + 1);
  //partitioned_tot_key_list.resize(tot_key_list.size());
  //partitioned_tot_valueptr_list.resize(tot_valueptr_list.size());

  // save the ev with kSavedPartitionNum piece of tensor
  // so that we can dynamically load ev with changed partition number
  bool save_unfiltered_features = true;
  TF_CHECK_OK(ReadBoolFromEnvVar(
      "TF_EV_SAVE_FILTERED_FEATURES", true, &save_unfiltered_features));
  int64 filter_freq = ev->MinFreq();
  for (size_t i = 0; i < tot_key_list.size(); i++) {
    for (int partid = 0; partid < kSavedPartitionNum; partid++) {
      if (tot_key_list[i] % kSavedPartitionNum == partid) {
        if (tot_valueptr_list[i] == reinterpret_cast<V*>(-1)) {
            // only forward, no backward, bypass
        } else if (tot_valueptr_list[i] == nullptr) {
          if (filter_freq != 0) {
            if (save_unfiltered_features) {
              key_filter_list_parts[partid].push_back(tot_key_list[i]);
            }
          } else {
            key_list_parts[partid].push_back(tot_key_list[i]);
            valueptr_list_parts[partid].push_back(
                ev->GetDefaultValue(tot_key_list[i]));
          }
        } else {
          key_list_parts[partid].push_back(tot_key_list[i]);
          valueptr_list_parts[partid].push_back(tot_valueptr_list[i]);
        }
        break;
      }
    }
  }

  for (size_t i = 0; i < tot_version_list.size(); i++) {
    for (int partid = 0; partid < kSavedPartitionNum; partid++) {
      if (tot_key_list[i] % kSavedPartitionNum == partid) {
        if (tot_valueptr_list[i] == reinterpret_cast<V*>(-1)) {
          // only forward, no backward, bypass
        } else if (tot_valueptr_list[i] == nullptr) {
          if (filter_freq != 0) {
            if (save_unfiltered_features) {
              version_filter_list_parts[partid].push_back(tot_version_list[i]);
            }
          } else {
            version_list_parts[partid].push_back(tot_version_list[i]);
          }
        } else {
          version_list_parts[partid].push_back(tot_version_list[i]);
        }
        break;
      }
    }
  }
  
  for (size_t i = 0; i < tot_freq_list.size(); i++) {
    for (int partid = 0; partid < kSavedPartitionNum; partid++) {
      if (tot_key_list[i] % kSavedPartitionNum == partid) {
        if (tot_valueptr_list[i] == reinterpret_cast<V*>(-1)) {
          // only forward, no backward, bypass
        } else if (tot_valueptr_list[i] == nullptr) {
          if (filter_freq != 0) {
            if (save_unfiltered_features) {
              freq_filter_list_parts[partid].push_back(tot_freq_list[i]);
            }
          } else {
            freq_list_parts[partid].push_back(tot_freq_list[i]);
          }
        } else {
          freq_list_parts[partid].push_back(tot_freq_list[i]);
        }
        break;
      }
    }
  }

  auto part_offset_flat = part_offset_tensor->flat<int32>();
  part_offset_flat(0) = 0;
  part_filter_offset[0] = 0;
  int ptsize = 0;
  for (int partid = 0; partid < kSavedPartitionNum; partid++) {
    std::vector<K>& key_list = key_list_parts[partid];
    std::vector<V* >& valueptr_list = valueptr_list_parts[partid];
    std::vector<int64>& version_list = version_list_parts[partid];
    std::vector<int64>& freq_list = freq_list_parts[partid];
    std::vector<K>& key_filter_list = key_filter_list_parts[partid];
    std::vector<int64>& version_filter_list =
      version_filter_list_parts[partid];
    std::vector<int64>& freq_filter_list = freq_filter_list_parts[partid];

    ptsize += key_list.size();
    for (int inpid = 0; inpid < key_list.size(); inpid++) {
      partitioned_tot_key_list.push_back(key_list[inpid]);
      partitioned_tot_valueptr_list.push_back(valueptr_list[inpid]);
    }
    for (int inpid = 0; inpid < version_list.size(); inpid++) {
      partitioned_tot_version_list.push_back(version_list[inpid]);
    }
    for (int inpid = 0; inpid < freq_list.size(); inpid++) {
      partitioned_tot_freq_list.push_back(freq_list[inpid]);
    }
    for (int inpid = 0; inpid < key_filter_list.size(); inpid++) {
      partitioned_tot_key_filter_list.push_back(key_filter_list[inpid]);
    }
    for (int inpid = 0; inpid < version_filter_list.size(); inpid++) {
      partitioned_tot_version_filter_list.push_back(version_filter_list[inpid]);
    }
    for (int inpid = 0; inpid < freq_filter_list.size(); inpid++) {
      partitioned_tot_freq_filter_list.push_back(freq_filter_list[inpid]);
    }

    part_offset_flat(partid + 1) = part_offset_flat(partid) + key_list.size();
    part_filter_offset[partid + 1] = part_filter_offset[partid] + key_filter_list.size();
  }
  // TODO: DB iterator not support partition_offset
  if (it != nullptr) {
    it->SetPartOffset((int32*)part_offset_tensor->data());
  }
  writer->Add(tensor_key + "-partition_offset", *part_offset_tensor);
  for(int i = 0; i <  kSavedPartitionNum + 1; i++) {
    part_offset_flat(i) = part_filter_offset[i];
  }
  if (it != nullptr) {
    it->SetPartFilterOffset((int32*)part_offset_tensor->data());
  }
  writer->Add(tensor_key + "-partition_filter_offset", *part_offset_tensor);

  VLOG(1) << "EV before partition:" << tensor_key << ", keysize:" <<  tot_key_list.size()
          << ", valueptr size:" << tot_valueptr_list.size();
  VLOG(1) << "EV after partition:" << tensor_key << ", ptsize:" << ptsize
          << ", keysize:"<<  partitioned_tot_key_list.size()
          <<", valueptr size:" << partitioned_tot_valueptr_list.size();

  size_t bytes_limit = 8 << 20;
  char* dump_buffer = (char*)malloc(sizeof(char) * bytes_limit);
  Status st;
  if (it != nullptr) {
    it->SwitchToAdmitFeatures();
  }
  EVKeyDumpIterator<K> ev_key_dump_iter(partitioned_tot_key_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-keys", writer, dump_buffer,
                                 bytes_limit, &ev_key_dump_iter,
                                 TensorShape({partitioned_tot_key_list.size() + iterator_size}),
                                 it);
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVValueDumpIterator<K, V> ev_value_dump_iter(ev, partitioned_tot_valueptr_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-values", writer, dump_buffer,
      bytes_limit, &ev_value_dump_iter,
      TensorShape({partitioned_tot_key_list.size() + iterator_size, ev->ValueLen()}),
      it, ev->storage()->GetOffset(ev->GetEmbeddingIndex()));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVVersionDumpIterator<int64> ev_version_dump_iter(partitioned_tot_version_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-versions", writer, dump_buffer,
      bytes_limit, &ev_version_dump_iter,
      TensorShape({partitioned_tot_version_list.size() + iterator_size}),
      it, -3);
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVFreqDumpIterator<int64> ev_freq_dump_iter(partitioned_tot_freq_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-freqs", writer, dump_buffer,
      bytes_limit, &ev_freq_dump_iter,
      TensorShape({partitioned_tot_freq_list.size() + iterator_size}),
      it, -2);
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }
  if (it != nullptr) {
    it->SwitchToFilteredFeatures();
  }
  EVKeyDumpIterator<K> ev_key_filter_dump_iter(partitioned_tot_key_filter_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-keys_filtered",
      writer, dump_buffer, bytes_limit, &ev_key_filter_dump_iter,
      TensorShape({partitioned_tot_key_filter_list.size()
          + filter_iterator_size}), it);
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVVersionDumpIterator<int64> ev_version_filter_dump_iter(
      partitioned_tot_version_filter_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-versions_filtered",
      writer, dump_buffer, bytes_limit, &ev_version_filter_dump_iter,
      TensorShape({partitioned_tot_version_filter_list.size()
          + filter_iterator_size}), it, -3);
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVFreqDumpIterator<int64> ev_freq_filter_dump_iter(
      partitioned_tot_freq_filter_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-freqs_filtered",
      writer, dump_buffer, bytes_limit, &ev_freq_filter_dump_iter,
      TensorShape({partitioned_tot_freq_filter_list.size()
          + filter_iterator_size}), it, -2);
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  free(dump_buffer);

  if (it != nullptr) {
    ev->storage()->iterator_mutex_unlock();
    delete it;
  }

  if (ev->IsSingleHbm() && tot_valueptr_list.size() > 0) {
    TypedAllocator::Deallocate(
        cpu_allocator(), tot_valueptr_list[0],
        tot_valueptr_list.size() * ev->ValueLen());
  }
  return Status::OK();
}

Status MoveMatchingFiles(
    Env* env,
    const tstring& pattern,
    const tstring& merged_prefix,
    int64 input_prefix_size);

/*Move two files and one directory:
1. xxxxx-ssd_record.index
2. xxxxx-ssd_record.data 
3. xxxxxx-emb_files/ 
1 and 2 record the meta data of SSDHash,
and 3 records the embeddings on SSD*/
Status MoveSsdFiles(Env* env,
    const gtl::ArraySlice<tstring>& input_prefixes,
    const tstring& merged_prefix);
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_

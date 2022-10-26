/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "parameters/raw_param.h"

#include <fstream>
#include <system_error>

#include "common.h"
#include "embeddings/embedding_layer.h"
#include "hashtable/identity_hashtable.h"
#include "hashtable/nv_hashtable.h"
#include "tensor_buffer/tensor2_wrapper.h"

namespace SparseOperationKit {

RawParam::RawParam(const std::string& initializer, const bool use_hashtable,
                   const std::vector<size_t> shape,
                   const std::shared_ptr<ResourcesManager>& resource_mgr,
                   const std::string var_name, const bool trainable)
    : resource_mgr_(resource_mgr),
      buffers_(resource_mgr->get_local_gpu_count(), nullptr),
      hashtables_(resource_mgr->get_local_gpu_count(), nullptr),
      max_vocabulary_size_per_gpu_(shape[0]),
      embedding_vector_size_(shape[1]),
      var_name_(var_name),
      trainable_(trainable),
      initializer_(Initializer::Get(initializer)),
      use_hashtable_(use_hashtable),
      initialized_(resource_mgr->get_local_gpu_count(), false) {
  emb_table_tensors_.reserve(resource_mgr_->get_local_gpu_count());
  emb_table_tensors_interface_.reserve(resource_mgr_->get_local_gpu_count());

  HugeCTR::CudaDeviceContext device_context;
  for (size_t dev_id = 0; dev_id < resource_mgr->get_local_gpu_count(); ++dev_id) {
    device_context.set_device(resource_mgr_->get_local_gpu(dev_id)->get_local_device_id());
    // create memory buffer
    buffers_[dev_id] = HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>::create();

    // reserve spaces for embedding table
    {
      Tensor2<float> tensor;
      buffers_[dev_id]->reserve(shape, &tensor);
      emb_table_tensors_.push_back(tensor);
      emb_table_tensors_interface_.push_back(Tensor2Wrapper<float>::create(tensor));
    }

    // construct hashtable
    {
      // TODO: make Types to be template
      if (use_hashtable_) {
        hashtables_[dev_id] = NvHashTable<int64_t, size_t>::create(max_vocabulary_size_per_gpu_);
      } else {
        hashtables_[dev_id] =
            IdentityHashTable<int64_t, size_t>::create(max_vocabulary_size_per_gpu_);
      }
    }
  }  // for dev_id

  if (emb_table_tensors_.size() != emb_table_tensors_interface_.size())
    throw std::runtime_error(ErrorBase +
                             "The size of embedding table tensors and its interface if not equal.");

  // allocate memory
  for (size_t dev_id = 0; dev_id < resource_mgr->get_local_gpu_count(); ++dev_id) {
    device_context.set_device(resource_mgr_->get_local_gpu(dev_id)->get_local_device_id());
    buffers_[dev_id]->allocate();
  }  // for dev_id
}

RawParam::~RawParam() {}

std::shared_ptr<RawParam> RawParam::create(const std::string& initializer, const bool use_hashtable,
                                           const std::vector<size_t> shape,
                                           const std::shared_ptr<ResourcesManager>& resource_mgr,
                                           const std::string var_name, const bool trainable) {
  return std::shared_ptr<RawParam>(
      new RawParam(initializer, use_hashtable, shape, resource_mgr, var_name, trainable));
}

size_t RawParam::get_max_vocabulary_size_per_gpu() const { return max_vocabulary_size_per_gpu_; }

size_t RawParam::get_embedding_vec_size() const { return embedding_vector_size_; }

bool RawParam::is_initialized(const size_t local_replica_id) const {
  return initialized_[local_replica_id];
}

void RawParam::init(const size_t global_replica_id) {
  const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
  if (is_initialized(local_replica_id)) return;

  MESSAGE("Variable: " + var_name_ + " on global_replica_id: " + std::to_string(global_replica_id) +
          " start initialization");
  if (local_replica_id >= emb_table_tensors_.size())
    throw std::runtime_error(ErrorBase +
                             "local_replica_id is out of the range of emb_table_tensors.size().");

  const auto& local_gpu = resource_mgr_->get_local_gpu(local_replica_id);
#ifdef SOK_ASYNC
  // TODO: necessary?? the underlying buffer might be modified by framework??
  CK_CUDA(cudaStreamSynchronize(local_gpu->get_framework_stream()));
#endif
  initializer_->fill(emb_table_tensors_interface_[local_replica_id], local_gpu->get_sm_count(),
                     local_gpu->get_variant_curand_gen(), local_gpu->get_stream());

  resource_mgr_->sync_gpu(local_replica_id);

  MESSAGE("Variable: " + var_name_ + " on global_replica_id: " + std::to_string(global_replica_id) +
          " initialization done.");
}

bool RawParam::trainable() const { return trainable_; }

void RawParam::set_user(std::shared_ptr<EmbeddingLayer>& embedding) { user_ = embedding; }

auto RawParam::get_hashtable(const size_t local_replica_id) -> std::shared_ptr<HashTable>& {
  return hashtables_[local_replica_id];
}

std::shared_ptr<Tensor>& RawParam::get_embedding_table_tensor(const size_t local_replica_id) {
  if (local_replica_id >= emb_table_tensors_.size())
    throw std::runtime_error(ErrorBase +
                             "local_replica_id is out of the range of emb_table_tensors.size().");

  return emb_table_tensors_interface_[local_replica_id];
}

std::string RawParam::get_var_name() const { return var_name_; }

void RawParam::set_initial_value(const size_t local_replica_id,
                                 const std::shared_ptr<Tensor>& initial_value) {
  auto& embedding_table = get_embedding_table_tensor(local_replica_id);
  if (embedding_table->get_num_elements() != initial_value->get_num_elements())
    throw std::runtime_error(ErrorBase +
                             "The number of elements in initial_value is different from that "
                             "of the embedding variable.");

  auto& local_gpu = resource_mgr_->get_local_gpu(local_replica_id);
#ifdef SOK_ASYNC
  CK_CUDA(cudaStreamSynchronize(local_gpu->get_framework_stream()));
#endif
  CK_CUDA(cudaMemcpyAsync(
      embedding_table->GetPtrWithType<float>(), initial_value->GetPtrWithType<float>(),
      initial_value->get_size_in_bytes(), cudaMemcpyDefault, local_gpu->get_stream()));
  CK_CUDA(cudaStreamSynchronize(local_gpu->get_stream()));

  initialized_[local_replica_id] = true;
  const size_t global_replica_id = resource_mgr_->cal_global_id_from_local_id(local_replica_id);
  MESSAGE("Variable: " + var_name_ + " on global_replica_id: " + std::to_string(global_replica_id) +
          " set initial_value.");
}

void RawParam::dump_to_file(const std::string filepath) {
  // step 1: allocate CPU spaces.
  auto host_buffer = HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>::create();
  HugeCTR::Tensor2<int64_t> keys;
  HugeCTR::Tensor2<float> embedding_values;
  if (0 == resource_mgr_->get_worker_id()) {
    // FIXME: what if there is not enough CPU memory ??
    const size_t total_max_vocabulary_size =
        get_max_vocabulary_size_per_gpu() * resource_mgr_->get_global_gpu_count();
    host_buffer->reserve({total_max_vocabulary_size}, &keys);
    host_buffer->reserve({total_max_vocabulary_size, get_embedding_vec_size()}, &embedding_values);
    host_buffer->allocate();
  }  // cheif-worker

  // step 2: let user to prepare keys & emebdding_values for all GPUs
  std::shared_ptr<Tensor> host_keys = Tensor2Wrapper<int64_t>::create(keys);
  std::shared_ptr<Tensor> host_embedding_values = Tensor2Wrapper<float>::create(embedding_values);
  size_t num_total_keys = 0ul;
  user_->save_params(host_keys, host_embedding_values, num_total_keys);
  resource_mgr_->sync_all_workers();

  // step 3: cheif worker dump those values to file.
  if (0 == resource_mgr_->get_worker_id()) {
    const std::string key_filename = filepath + "/" + var_name_ + "_keys.file";
    const std::string values_filename = filepath + "/" + var_name_ + "_values.file";
    std::ofstream key_stream(key_filename, std::ios::binary | std::ios::out);
    std::ofstream values_stream(values_filename, std::ios::binary | std::ios::out);
    if (!key_stream.is_open())
      throw std::runtime_error(ErrorBase + "Cannot open " + key_filename + " for writing.");
    if (!values_stream.is_open())
      throw std::runtime_error(ErrorBase + "Cannot open" + values_filename + " for writing.");
    key_stream.write(reinterpret_cast<char*>(keys.get_ptr()), sizeof(int64_t) * num_total_keys);
    values_stream.write(reinterpret_cast<char*>(embedding_values.get_ptr()),
                        sizeof(float) * num_total_keys * get_embedding_vec_size());
    key_stream.close();
    values_stream.close();
  }  // cheif worker

  // step 4: synchronize all workers.
  resource_mgr_->sync_all_workers();
}

void RawParam::let_user_dump_to_file(const std::string filepath) { user_->dump_to_file(filepath); }

void RawParam::restore_from_file(const std::string filepath) {
  const std::string key_filename = filepath + "/" + var_name_ + "_keys.file";
  const std::string values_filename = filepath + "/" + var_name_ + "_values.file";
  if (!file_exist(key_filename))
    throw std::runtime_error(ErrorBase + key_filename +
                             " doesn't exist. "
                             "This is probably because the keys of the EmbeddingVariable"
                             " was not stored during the dump_to_file.");
  if (!file_exist(values_filename))
    throw std::runtime_error(ErrorBase + values_filename +
                             " doesn't exist. "
                             "This is probably because the values of the EmbeddingVariable"
                             " was not stored during the dump_to_file.");

  std::ifstream key_stream(key_filename, std::ios::binary | std::ios::in);
  std::ifstream values_stream(values_filename, std::ios::binary | std::ios::in);

  // step 1: check whether the number of content is consistent and valid.
  key_stream.seekg(0, key_stream.end);
  const size_t key_size_in_bytes = key_stream.tellg();
  key_stream.seekg(0, key_stream.beg);

  values_stream.seekg(0, values_stream.end);
  const size_t values_size_in_bytes = values_stream.tellg();
  values_stream.seekg(0, values_stream.beg);

  if (key_size_in_bytes == 0 || values_size_in_bytes == 0)
    throw std::runtime_error(ErrorBase +
                             "Invalid files, several file(s) size is 0 bytes, where "
                             "key: " +
                             std::to_string(key_size_in_bytes) +
                             "bytes, values: " + std::to_string(values_size_in_bytes) + "bytes.");
  if (key_size_in_bytes % sizeof(int64_t) != 0)
    throw std::runtime_error(ErrorBase +
                             "Invalid file stream for keys, because the count of "
                             "keys is not divisible by sizeof(int64).");
  if (values_size_in_bytes % (sizeof(float) * embedding_vector_size_) != 0)
    throw std::runtime_error(ErrorBase +
                             "Invalid file stream for embedding values, because the "
                             "count of embedding values is not divisible by "
                             "sizeof(float) * embedding_vector_size.");
  const size_t key_count = key_size_in_bytes / sizeof(int64_t);
  const size_t values_count = values_size_in_bytes / (sizeof(float) * embedding_vector_size_);
  if (key_count ^ values_count)
    throw std::runtime_error(ErrorBase +
                             "The count of key and values are not consistent, which are "
                             "key: " +
                             std::to_string(key_count) +
                             ", values: " + std::to_string(values_count) + " respectively.");

  // step 2: allocate temp spaces
  std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>> host_buffer =
      HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>::create();
  HugeCTR::Tensor2<int64_t> keys;
  host_buffer->reserve({key_count}, &keys);
  HugeCTR::Tensor2<float> embedding_values;
  host_buffer->reserve({values_count, embedding_vector_size_}, &embedding_values);
  host_buffer->allocate();
  MESSAGE("Allocated temporary pinned buffer for loading parameters.");

  // step 3: read content from file to pinned memory
  key_stream.read(reinterpret_cast<char*>(keys.get_ptr()), key_size_in_bytes);
  values_stream.read(reinterpret_cast<char*>(embedding_values.get_ptr()), values_size_in_bytes);

  key_stream.close();
  values_stream.close();

  // step 4: upload content to GPU memory
  // because how to load parameters to each GPU is related to
  // how will the embedding lookuper use those parameters.
  // so that delegate this loading job to embedding lookuper
  user_->restore_params(/*keys=*/Tensor2Wrapper<int64_t>::create(keys),
                        /*embedding_values=*/Tensor2Wrapper<float>::create(embedding_values),
                        /*num_total_keys=*/key_count);

  // finnaly: synchronize all workers.
  resource_mgr_->sync_all_workers();
}

void RawParam::let_user_restore_from_file(const std::string filepath) {
  user_->restore_from_file(filepath);
}

void RawParam::load_embedding_values(const std::vector<std::shared_ptr<Tensor>>& tensor_list) {
  // step 1: allocate temp spaces
  size_t total_key_count = 0;
  for (const auto& tensor : tensor_list) {
    total_key_count += (tensor->get_num_elements() / embedding_vector_size_);
  }  // iter on tensors

  std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>> host_buffer =
      HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>::create();

  Tensor2<int64_t> keys;
  host_buffer->reserve({total_key_count}, &keys);
  Tensor2<float> embedding_values;
  host_buffer->reserve({total_key_count, embedding_vector_size_}, &embedding_values);
  host_buffer->allocate();
  MESSAGE("Allocated temporary buffer for loading embedding values.");

  // step 2: generate keys and copy content to temp spaces.
  for (size_t i = 0; i < total_key_count; i++) keys.get_ptr()[i] = static_cast<int64_t>(i);
  size_t offset = 0;
  for (const auto& tensor : tensor_list) {
    std::memcpy(embedding_values.get_ptr() + offset, tensor->GetPtrWithType<float>(),
                tensor->get_size_in_bytes());
    offset += tensor->get_num_elements();
  }  // iter on tensors
  if (embedding_values.get_num_elements() != offset)
    throw std::runtime_error(ErrorBase + "Error happened in copy tensor content.");

  // step 3: upload content to GPU memory
  // because how to load parameters to each GPU is related to
  // how will the embedding lookuper use those parameters.
  // so delegate this loading job to embedding lookuper
  user_->restore_params(/*keys=*/Tensor2Wrapper<int64_t>::create(keys),
                        /*embedding_values=*/Tensor2Wrapper<float>::create(embedding_values),
                        /*num_total_keys=*/total_key_count);

  resource_mgr_->sync_all_workers();
}

void RawParam::let_user_load_embedding_values(
    const std::vector<std::shared_ptr<Tensor>>& tensor_list) {
  user_->load_embedding_values(tensor_list);
}

void RawParam::set_hashtable(std::shared_ptr<BaseSimpleHashtable> hashtable) {
  for (size_t local_replica_id = 0ul; local_replica_id < resource_mgr_->get_local_gpu_count();
       ++local_replica_id) {
    const size_t global_replica_id = resource_mgr_->cal_global_id_from_local_id(local_replica_id);
    auto temp_hashtable = hashtable->clone(global_replica_id);
    hashtables_[local_replica_id] = temp_hashtable;
  }
}

}  // namespace SparseOperationKit
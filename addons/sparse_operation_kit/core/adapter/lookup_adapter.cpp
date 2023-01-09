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

#include "lookup_adapter.hpp"

#include "common/check.h"

#ifdef GOOGLE_CUDA
#ifdef TENSORFLOW_USE_GPU_EV

namespace tensorflow {

template <typename KeyType, typename DType>
void EmbeddingVarGPUAdapter<KeyType, DType>::set(
    OpKernelContext* ctx,
    std::vector<core::RefCountPtr<EmbeddingVarGPU<KeyType, DType>>>& vars,
    const std::vector<int>& ev_size_per_lookup, cudaStream_t stream) {
  vars_.resize(vars.size());
  for (int i = 0; i < vars.size(); ++i) {
    vars_[i] = vars[i].get();
  }
  ctx_ = ctx;
  ev_size_per_lookup_ = ev_size_per_lookup;
  stream_ = stream;
}

template <typename KeyType, typename DType>
void EmbeddingVarGPUAdapter<KeyType, DType>::lookup(
    const ::core::Tensor& keys, size_t num_keys,
    const ::core::Tensor& id_space_offset, size_t num_id_space_offset,
    const ::core::Tensor& id_space, ::core::TensorList& embedding_vec) {
  id_space_offset_.resize(num_id_space_offset);
  CUDACHECK(cudaMemcpyAsync(id_space_offset_.data(),
                            id_space_offset.get<uint32_t>(),
                            sizeof(uint32_t) * (num_id_space_offset),
                            cudaMemcpyDeviceToHost, stream_));
  id_space_.resize(num_id_space_offset - 1);
  CUDACHECK(cudaMemcpyAsync(id_space_.data(), id_space.get<int>(),
                            sizeof(int) * (num_id_space_offset - 1),
                            cudaMemcpyDeviceToHost, stream_));
  CUDACHECK(cudaStreamSynchronize(stream_));
  assert(tmp_ev_list_.size() == 0);

  const KeyType* input = keys.get<KeyType>();
  std::vector<DType*> lookup_res;
  for (int i = 0; i < num_id_space_offset - 1; ++i) {
    size_t num = id_space_offset_[i + 1] - id_space_offset_[i];

    if (num == 0) {
      continue;
    }
    int ev_size = ev_size_per_lookup_[i];
    Tensor evs;
    OP_REQUIRES_OK(ctx_, ctx_->allocate_temp(DT_FLOAT, {ev_size * num}, &evs));
    tmp_ev_list_.push_back(evs);

    const auto& device = ctx_->eigen_device<Eigen::GpuDevice>();
    auto var = vars_[id_space_[i]];
    var->LookupOrCreate(input + id_space_offset_[i], evs.flat<DType>().data(),
                        var->GetDefaultValuePtr(), var->GetDefaultValueDim(),
                        true, num, device);
    for (size_t i_ev = 0; i_ev < num; ++i_ev) {
      lookup_res.push_back(evs.flat<DType>().data() + i_ev * ev_size);
    }
  }
  DType** output = embedding_vec.get<DType>();
  CUDACHECK(cudaMemcpyAsync(output, lookup_res.data(),
                            sizeof(DType*) * lookup_res.size(),
                            cudaMemcpyHostToDevice, stream_));
  CUDACHECK(cudaStreamSynchronize(stream_));
}

template class EmbeddingVarGPUAdapter<int32, float>;
// template class EmbeddingVarGPUAdapter<int32_t, __half>;
template class EmbeddingVarGPUAdapter<int64, float>;
// template class EmbeddingVarGPUAdapter<int64_t, __half>;
}  // namespace tensorflow
#endif
#endif
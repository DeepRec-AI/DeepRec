/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#pragma once

// clang-format off
#include <cuda_fp16.h>

#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"

#include "HugeCTR/embedding/embedding_table.hpp"
#include "tensorflow/core/framework/embedding/embedding_var.h"

// clang-format on

#ifdef GOOGLE_CUDA

namespace tensorflow {
namespace core23 = HugeCTR::core23;

template <typename KeyType, typename OffsetType, typename DType>
class EmbeddingVarGPUAdapter : public ::embedding::ILookup {
 public:
  virtual ~EmbeddingVarGPUAdapter() = default;

  void set(
      OpKernelContext* ctx,
      std::vector<core::RefCountPtr<EmbeddingVar<KeyType, DType>>>& vars,
      const std::vector<int>& ev_size_per_lookup, cudaStream_t stream);

  void lookup(const core23::Tensor& keys, size_t num_keys,
              const core23::Tensor& id_space_offset, size_t num_id_space_offset,
              const core23::Tensor& id_space,
              core23::Tensor& embedding_vec) override;

  void clear_tmp_ev_list() { tmp_ev_list_.clear(); }

 private:
  std::vector<EmbeddingVar<KeyType, DType>*> vars_;
  std::vector<OffsetType> id_space_offset_;
  std::vector<int> id_space_;
  cudaStream_t stream_;
  OpKernelContext* ctx_;
  std::vector<int> ev_size_per_lookup_;
  std::vector<Tensor> tmp_ev_list_;
};

}  // namespace tensorflow
#endif

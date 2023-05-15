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
=======================================================================*/

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"

namespace tensorflow {

namespace {

template <typename TKey, typename TValue>
struct GroupEmbeddingBackWardArgs {
  GroupEmbeddingBackWardArgs() = default;
  GroupEmbeddingBackWardArgs(TValue *grads, TKey *sp_values,
                             TValue *emb_variable, TValue *grads_output,
                             int *offset_indices, int nnz)
      : grads_(grads), sp_values_(sp_values),emb_variable_(emb_variable),
        grads_output_(grads_output), offset_indices_(offset_indices), nnz_(nnz)  {}
  TValue *grads_;
  TKey *sp_values_;
  TValue *emb_variable_;
  TValue *grads_output_;
  int *offset_indices_;
  int nnz_;
};

template <typename TKey, typename TValue, Combiner combiner, int Tilesize>
__global__ void ComputeEVGradFn(
    const int batch_size, const float max_norm, const int num_lookups,
    const int dimension, GroupEmbeddingBackWardArgs<TKey, TValue> *args) {
  float l2_sum;

  const auto &block = cooperative_groups::this_thread_block();
  const auto &tile = cooperative_groups::tiled_partition<Tilesize>(block);
  // each block partition corresponding to one sample
  const int bid =
      block.group_index().x * tile.meta_group_size() + tile.meta_group_rank();
  // each thread corresponding to one element in the embedding vector
  const int tid = tile.thread_rank();

  if (bid < batch_size && tid < dimension) {
    for (int idx = 0; idx < num_lookups; ++idx) {
      int value_offset = args[idx].offset_indices_[bid];
      int feature_num;
      if (bid == (batch_size - 1)) {
        feature_num = args[idx].nnz_ - value_offset;
      } else {
        feature_num = args[idx].offset_indices_[bid + 1] - value_offset;
      }

      if (feature_num > 0) {
        float grad = args[idx].grads_[bid * dimension + tid];
        grad = CombineGrad<combiner>(grad, feature_num);

        for (int j = 0; j < feature_num; ++j) {
          float grad_i = grad;
          int feature_offset = (value_offset + j) * dimension;
          if (max_norm > 0.0f) {
            float emb_element = 0.0f;  // TODO(junqihu): get emb_weight
            if (tid == 0) {
              l2_sum = 0.0f;
            }
            tile.shfl(l2_sum, 0);
            atomicAdd(&l2_sum, emb_element * emb_element);
            tile.sync();
            float l2_norm = sqrtf(l2_sum);
            if (l2_norm > max_norm) {
              grad_i *= max_norm / l2_norm;
            }
          }
          args[idx].grads_output_[(value_offset + j) * dimension + tid] =
              grad_i;
        }
      }
    }
  }
}

template <typename TKey, typename TValue, Combiner combiner, int Tilesize>
__global__ void ComputeSparseGradFn(
    const int batch_size, const float max_norm, const int num_lookups,
    const int dimension, GroupEmbeddingBackWardArgs<TKey, TValue> *args) {
  float l2_sum;
  const auto &block = cooperative_groups::this_thread_block();
  const auto &tile = cooperative_groups::tiled_partition<Tilesize>(block);
  // each block partition corresponding to one sample
  const int bid =
      block.group_index().x * tile.meta_group_size() + tile.meta_group_rank();
  // each thread corresponding to one element in the embedding vector
  const int tid = tile.thread_rank();

  for (int idx = 0; idx < num_lookups; ++idx) {
    const int value_offset = args[idx].offset_indices_[bid];
    int feature_num;
    if (bid == (batch_size - 1)) {
      feature_num = args[idx].nnz_ - value_offset;
    } else {
      feature_num = args[idx].offset_indices_[bid + 1] - value_offset;
    }

    if (feature_num > 0) {
      float grad = args[idx].grads_[bid * dimension + tid];
      grad = CombineGrad<combiner>(grad, feature_num);
      for (int i = 0; i < feature_num; i++) {
        float grad_i = grad;
        if (max_norm > 0.0f) {
          int64_t indices = int(args[idx].sp_values_[value_offset + i]);
          float emb_element =
              args[idx].emb_variable_[indices * dimension + tid];
          if (tid == 0) {
            l2_sum = 0.0f;
          }
          tile.shfl(l2_sum, 0);
          atomicAdd(&l2_sum, emb_element * emb_element);
          tile.sync();
          float l2_norm = sqrtf(l2_sum);
          if (l2_norm > max_norm) {
            grad_i *= max_norm / l2_norm;
          }
        }
        args[idx].grads_output_[(value_offset + i) * dimension + tid] = grad_i;
      }
    }
  }
}

template <typename TKey, typename TValue, Combiner combiner>
__global__ void NormalComputeEVGradFn(
    const int batch_size, const float max_norm, const int num_lookups,
    const int dimension, GroupEmbeddingBackWardArgs<TKey, TValue> *args) {
  __shared__ TValue l2_sum[1];

  const auto &block = cooperative_groups::this_thread_block();
  // each block partition corresponding to one sample
  const int bid = block.group_index().x;
  // each thread corresponding to one element in the embedding vector
  const int tid = block.thread_rank();

  if (bid < batch_size && tid < dimension) {
    for (int idx = 0; idx < num_lookups; ++idx) {
      int value_offset = args[idx].offset_indices_[bid];
      int feature_num;
      if (bid == (batch_size - 1)) {
        feature_num = args[idx].nnz_ - value_offset;
      } else {
        feature_num = args[idx].offset_indices_[bid + 1] - value_offset;
      }

      if (feature_num > 0) {
        float grad = args[idx].grads_[bid * dimension + tid];
        grad = CombineGrad<combiner>(grad, feature_num);

        for (int j = 0; j < feature_num; ++j) {
          float grad_i = grad;
          int feature_offset = (value_offset + j) * dimension;
          if (max_norm > 0.0f) {
            float emb_element = 0.0f;  // TODO(junqihu): get emb_weight
            if (tid == 0) {
              l2_sum[0] = 0.0f;
            }
            __syncthreads();
            atomicAdd(l2_sum, emb_element * emb_element);
            __syncthreads();
            float l2_norm = sqrtf(l2_sum[0]);
            if (l2_norm > max_norm) {
              grad_i *= max_norm / l2_norm;
            }
          }
          args[idx].grads_output_[(value_offset + j) * dimension + tid] =
              grad_i;
        }
      }
    }
  }
}

template <typename TKey, typename TValue, Combiner combiner>
__global__ void NormalComputeSparseGradFn(
    const int batch_size, const float max_norm, const int num_lookups,
    const int dimension, GroupEmbeddingBackWardArgs<TKey, TValue> *args) {
  __shared__ TValue l2_sum[1];

  const auto &block = cooperative_groups::this_thread_block();
  // each block partition corresponding to one sample
  const int bid = block.group_index().x;
  // each thread corresponding to one element in the embedding vector
  const int tid = block.thread_rank();

  for (int idx = 0; idx < num_lookups; ++idx) {
    const int value_offset = args[idx].offset_indices_[bid];
    int feature_num;
    if (bid == (batch_size - 1)) {
      feature_num = args[idx].nnz_ - value_offset;
    } else {
      feature_num = args[idx].offset_indices_[bid + 1] - value_offset;
    }

    if (feature_num > 0) {
      float grad = args[idx].grads_[bid * dimension + tid];
      grad = CombineGrad<combiner>(grad, feature_num);
      for (int i = 0; i < feature_num; i++) {
        float grad_i = grad;
        if (max_norm > 0.0f) {
          int64_t indices = int(args[idx].sp_values_[value_offset + i]);
          float emb_element =
              args[idx].emb_variable_[indices * dimension + tid];
          if (tid == 0) {
            l2_sum[0] = 0.0f;
          }
          __syncthreads();
          atomicAdd(l2_sum, emb_element * emb_element);
          __syncthreads();
          float l2_norm = sqrtf(l2_sum[0]);
          if (l2_norm > max_norm) {
            grad_i *= max_norm / l2_norm;
          }
        }
        args[idx].grads_output_[(value_offset + i) * dimension + tid] = grad_i;
      }
    }
  }
}

}  // namespace

template <typename TKey, typename TValue>
class GroupEmbeddingLookupBackWard {
 public:
  explicit GroupEmbeddingLookupBackWard(int dimension, int num_lookups,
                                        float max_norm,
                                        Allocator *gpu_allocator = nullptr)
      : alloc_(gpu_allocator) {
    d_args_ =
        TypedAllocator::Allocate<GroupEmbeddingBackWardArgs<TKey, TValue>>(
            gpu_allocator, num_lookups, AllocationAttributes());
    h_args_.reserve(num_lookups);
    max_norm_ = max_norm;
    nums_ = num_lookups;
    dimension_ = dimension;
  }

  void set(GroupEmbeddingBackWardArgs<TKey, TValue> &arg) {
    h_args_.emplace_back(arg);
  }

  ~GroupEmbeddingLookupBackWard() {
    TypedAllocator::Deallocate(alloc_, d_args_, nums_);
  }

  template <typename GradFn>
  inline void Backward(GradFn fn, int batch_size, int tile_size,
                       cudaStream_t stream) {
    CK_CUDA_THROW_(cudaMemcpyAsync(
        d_args_, h_args_.data(),
        h_args_.size() * sizeof(GroupEmbeddingBackWardArgs<TKey, TValue>),
        cudaMemcpyHostToDevice, stream));

    {
      if (tile_size <= 32) {
        const int block_size = batch_size / 64 * tile_size + 1;

        fn<<<block_size, 64, 0, stream>>>(batch_size, max_norm_, nums_,
                                          dimension_, d_args_);
      } else {
        fn<<<batch_size, tile_size, 0, stream>>>(batch_size, max_norm_, nums_,
                                                 dimension_, d_args_);
      }
    }

    CK_CUDA_THROW_(cudaGetLastError());
  }

 protected:
  std::vector<GroupEmbeddingBackWardArgs<TKey, TValue>> h_args_;
  GroupEmbeddingBackWardArgs<TKey, TValue> *d_args_;
  Allocator *alloc_;
  float max_norm_;
  int nums_;
  int dimension_;
};

template <typename TKey, typename TValue>
class GroupLookupBackWardBaseOp : public OpKernel {
 public:
  explicit GroupLookupBackWardBaseOp(OpKernelConstruction *c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
  }

  template <bool Isev = false, Combiner combiner>
  inline void compute(GroupEmbeddingLookupBackWard<TKey, TValue> &lookuper,
                      const int batch_size, cudaStream_t stream) {
    if (Isev) {
      if (dimension_ <= 2) {
        lookuper.Backward(ComputeEVGradFn<TKey, TValue, combiner, 2>,
                          batch_size, 2, stream);
      } else if (dimension_ <= 4) {
        lookuper.Backward(ComputeEVGradFn<TKey, TValue, combiner, 4>,
                          batch_size, 4, stream);
      } else if (dimension_ <= 8) {
        lookuper.Backward(ComputeEVGradFn<TKey, TValue, combiner, 8>,
                          batch_size, 8, stream);
      } else if (dimension_ <= 16) {
        lookuper.Backward(ComputeEVGradFn<TKey, TValue, combiner, 16>,
                          batch_size, 16, stream);
      } else if (dimension_ <= 32) {
        lookuper.Backward(ComputeEVGradFn<TKey, TValue, combiner, 32>,
                          batch_size, 32, stream);
      } else {
        lookuper.Backward(NormalComputeEVGradFn<TKey, TValue, combiner>,
                          batch_size, dimension_, stream);
      }
    } else {
      if (dimension_ <= 2) {
        lookuper.Backward(ComputeSparseGradFn<TKey, TValue, combiner, 2>,
                          batch_size, 2, stream);
      } else if (dimension_ <= 4) {
        lookuper.Backward(ComputeSparseGradFn<TKey, TValue, combiner, 4>,
                          batch_size, 4, stream);
      } else if (dimension_ <= 8) {
        lookuper.Backward(ComputeSparseGradFn<TKey, TValue, combiner, 8>,
                          batch_size, 8, stream);
      } else if (dimension_ <= 16) {
        lookuper.Backward(ComputeSparseGradFn<TKey, TValue, combiner, 16>,
                          batch_size, 16, stream);
      } else if (dimension_ <= 32) {
        lookuper.Backward(ComputeSparseGradFn<TKey, TValue, combiner, 32>,
                          batch_size, 32, stream);
      } else {
        lookuper.Backward(NormalComputeSparseGradFn<TKey, TValue, combiner>,
                          batch_size, dimension_, stream);
      }
    }
  }

 protected:
  std::string combiner_;
  float max_norm_;
  int num_lookups_;
  int dimension_;
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
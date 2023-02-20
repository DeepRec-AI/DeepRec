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

#include <string>

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename TKey, typename TValue>
struct GroupEmbeddingBackWardArgs {
  int nnz_;
  int64_t emb_row_size_;
  int *offset_indices_;
  TValue *emb_variable_;
  TValue *grads_;
  TValue *grads_output_;
  TKey *sp_values_;
};

template <typename TKey, typename TValue, Combiner combiner>
__global__ void ComputeEVGradFn(
    const int batch_size, const int dimension, const float max_norm,
    const int num_lookups, GroupEmbeddingBackWardArgs<TKey, TValue> *args) {
  __shared__ float l2_sum[1];

  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the
                          // embedding vector

  if (bid < batch_size && tid < dimension) {
    for (int idx = 0; idx < num_lookups; ++idx) {
      int value_offset = args[idx].offset_indices_[bid];
      int feature_num;
      if (blockIdx.x == int(batch_size) - 1) {
        feature_num = args[idx].nnz_ - value_offset;
      } else {
        feature_num = args[idx].offset_indices_[bid + 1] - value_offset;
      }
      float grad = args[idx].grads_[bid * dimension + tid];
      grad = CombineGrad<combiner>(grad, feature_num);

      for (int j = 0; j < feature_num; ++j) {
        float grad_i = grad;
        int feature_offset = (value_offset + j) * dimension;
        if (max_norm > 0.0f) {
          float emb_element = 0.0f;  // TODO: hujunqi get emb_weight
          if (threadIdx.x == 0) {
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
        args[idx].grads_output_[(value_offset + j) * dimension + threadIdx.x] =
            grad_i;
      }
    }
  }
}

template <typename TKey, typename TValue, Combiner combiner>
__global__ void ComputeSparseGradFn(
    const int batch_size, const int dimension, const float max_norm,
    const int num_lookups, GroupEmbeddingBackWardArgs<TKey, TValue> *args) {
  __shared__ float l2_sum[1];
  for (int idx = 0; idx < num_lookups; ++idx) {
    const int value_offset = args[idx].offset_indices_[blockIdx.x];
    int feature_num;
    if (blockIdx.x == int(batch_size) - 1) {
      feature_num = int(args[idx].nnz_) - value_offset;
    } else {
      feature_num = args[idx].offset_indices_[blockIdx.x + 1] - value_offset;
    }
    float grad = args[idx].grads_[blockIdx.x * dimension + threadIdx.x];
    // printf("feature_num is %d , grad is %lld , bid is %d , tid is %d \n",
    // feature_num, grad, blockIdx.x, threadIdx.x);
    grad = CombineGrad<combiner>(grad, feature_num);
    for (int i = 0; i < feature_num; i++) {
      float grad_i = grad;
      if (max_norm > 0.0f) {
        int64_t indices = int(args[idx].sp_values_[value_offset + i]);
        float emb_element = 0.0f;
        if (FastBoundsCheck(indices, args[idx].emb_row_size_)) {
          emb_element =
              args[idx].emb_variable_[indices * dimension + threadIdx.x];
        }
        emb_element =
            args[idx].emb_variable_[indices * dimension + threadIdx.x];
        if (threadIdx.x == 0) {
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
      args[idx].grads_output_[(value_offset + i) * dimension + threadIdx.x] =
          grad_i;
    }
  }
}

}  // namespace

template <typename TKey, typename TValue>
class GroupEmbeddingBackWard {
 public:
  void initialize(int num_lookups) {
    nums_ = num_lookups;
    args_.resize(num_lookups);
    CK_CUDA_THROW_(cudaMalloc(
        &d_args_,
        sizeof(GroupEmbeddingBackWardArgs<TKey, TValue>) * num_lookups));
  }

  void set(int idx, TValue *grads, TValue *grads_output, int *offset_indices,
           TKey *sp_values, TValue *emb_variable, int64_t emb_row_size,
           int nnz) {
    args_[idx].grads_ = grads;
    args_[idx].grads_output_ = grads_output;
    args_[idx].offset_indices_ = offset_indices;
    args_[idx].sp_values_ = sp_values;
    args_[idx].emb_variable_ = emb_variable;
    args_[idx].emb_row_size_ = emb_row_size;
    args_[idx].nnz_ = nnz;
  }

  ~GroupEmbeddingBackWard() {
    if (d_args_) {
      CK_CUDA_THROW_(cudaFree(d_args_));
    }
  }

  template <typename GradFn>
  void compute(GradFn fn, int batch_size, int dimension, float max_norm,
               cudaStream_t stream) {
    CK_CUDA_THROW_(cudaMemcpyAsync(
        d_args_, args_.data(),
        args_.size() * sizeof(GroupEmbeddingBackWardArgs<TKey, TValue>),
        cudaMemcpyHostToDevice, stream));
    CK_CUDA_THROW_(cudaStreamSynchronize(stream));
    {
      const int blocks = int(batch_size);
      const int threads = int(dimension);

      fn<<<blocks, threads, 0, stream>>>(batch_size, dimension, max_norm, nums_,
                                         d_args_);
    }

    CK_CUDA_THROW_(cudaGetLastError());
  }

 protected:
  int nums_;
  std::vector<GroupEmbeddingBackWardArgs<TKey, TValue>> args_;
  GroupEmbeddingBackWardArgs<TKey, TValue> *d_args_;
};

template <typename TKey, typename TValue>
class GroupLookupBackWardBaseOp : public OpKernel {
 public:
  explicit GroupLookupBackWardBaseOp(OpKernelConstruction *c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    lookuper_.initialize(num_lookups_);
  }

 protected:
  std::string combiner_;
  float max_norm_;
  int num_lookups_;
  int dimension_;
  GroupEmbeddingBackWard<TKey, TValue> lookuper_;
};

template <typename TFKey, typename TKey, typename TValue>
class MultiEmbeddingSparseLookupBackWardOp
    : public GroupLookupBackWardBaseOp<TKey, TValue> {
 public:
  explicit MultiEmbeddingSparseLookupBackWardOp(OpKernelConstruction *c)
      : GroupLookupBackWardBaseOp<TKey, TValue>(c) {}

  void Compute(OpKernelContext *ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    int batch_size = -1;
    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor grads_tensor = ctx->input(i);
      const Tensor emb_variables_tensor = ctx->input(this->num_lookups_ + i);
      const Tensor sp_values_tensor = ctx->input(2 * this->num_lookups_ + i);
      const Tensor sp_values_offset_tensor =
          ctx->input(3 * this->num_lookups_ + i);

      int64 emb_row_size = emb_variables_tensor.shape().dim_size(0);
      if (batch_size == -1) {
        batch_size = sp_values_offset_tensor.shape().dim_size(0);
      }

      int dimension = emb_variables_tensor.shape().dim_size(1);
      OP_REQUIRES(
          ctx, dimension == this->dimension_,
          errors::InvalidArgument(
              "shape[0] of each tensor in offset_indices are different."));

      const int64_t nnz = sp_values_tensor.NumElements();

      Tensor *grads_sp_values_tensor;
      TensorShape grads_sp_values_tensor_shape =
          TensorShape(std::vector<int64>({nnz, dimension}));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, grads_sp_values_tensor_shape,
                                               &grads_sp_values_tensor));
      this->lookuper_.set(
          i, const_cast<TValue *>(grads_tensor.flat<TValue>().data()),
          const_cast<TValue *>(grads_sp_values_tensor->flat<TValue>().data()),
          const_cast<int *>(sp_values_offset_tensor.flat<int>().data()),
          const_cast<TKey *>(reinterpret_cast<const TKey *>(
              sp_values_tensor.flat<TFKey>().data())),
          const_cast<TValue *>(emb_variables_tensor.flat<TValue>().data()),
          emb_row_size, nnz);
    }

    if (this->combiner_ == "mean") {
      this->lookuper_.compute(ComputeSparseGradFn<TKey, TValue, Mean>,
                              batch_size, this->dimension_, this->max_norm_,
                              stream);
    } else if (this->combiner_ == "sum") {
      this->lookuper_.compute(ComputeSparseGradFn<TKey, TValue, Sum>,
                              batch_size, this->dimension_, this->max_norm_,
                              stream);
    } else {
      this->lookuper_.compute(ComputeSparseGradFn<TKey, TValue, Sqrtn>,
                              batch_size, this->dimension_, this->max_norm_,
                              stream);
    }
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype) \
  REGISTER_KERNEL_BUILDER(                                 \
      Name("MultiEmbeddingSparseLookUpGrad")               \
          .Device(DEVICE_GPU)                              \
          .TypeConstraint<key_type_tf>("Tkeys")            \
          .TypeConstraint<dtype>("dtype"),                 \
      MultiEmbeddingSparseLookupBackWardOp<key_type_tf, key_type, dtype>)

REGISTER_GPU_KERNELS(int64, int64_t, float);
REGISTER_GPU_KERNELS(int32, int32_t, float);
#undef REGISTER_GPU_KERNELS

template <typename TFKey, typename TKey, typename TValue>
class MultiKvResourceGatherBackWardOp
    : public GroupLookupBackWardBaseOp<TKey, TValue> {
 public:
  explicit MultiKvResourceGatherBackWardOp(OpKernelConstruction *c)
      : GroupLookupBackWardBaseOp<TKey, TValue>(c) {}

  void Compute(OpKernelContext *ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    int batch_size = -1;
    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor grads_tensor = ctx->input(i);
      EmbeddingVar<TFKey, TValue> *ev = nullptr;
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, this->num_lookups_ + i),
                              &ev));
      core::ScopedUnref unref_me(ev);
      const Tensor sp_values_tensor = ctx->input(2 * this->num_lookups_ + i);
      const Tensor sp_values_offset_tensor =
          ctx->input(3 * this->num_lookups_ + i);
      int dimension = ev->ValueLen();
      if (batch_size == -1) {
        batch_size = sp_values_offset_tensor.shape().dim_size(0);
      }
      OP_REQUIRES(
          ctx, dimension == this->dimension_,
          errors::InvalidArgument(
              "shape[0] of each tensor in offset_indices are different."));

      const int64_t nnz = sp_values_tensor.NumElements();

      Tensor *grads_sp_values_tensor;
      TensorShape grads_sp_values_tensor_shape =
          TensorShape(std::vector<int64>({nnz, dimension}));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, grads_sp_values_tensor_shape,
                                               &grads_sp_values_tensor));
      this->lookuper_.set(
          i, const_cast<TValue *>(grads_tensor.flat<TValue>().data()),
          const_cast<TValue *>(grads_sp_values_tensor->flat<TValue>().data()),
          const_cast<int *>(sp_values_offset_tensor.flat<int>().data()),
          const_cast<TKey *>(reinterpret_cast<const TKey *>(
              sp_values_tensor.flat<TFKey>().data())),
          const_cast<TValue *>(grads_sp_values_tensor->flat<TValue>().data()),
          -1, nnz);
    }

    if (this->combiner_ == "mean") {
      this->lookuper_.compute(ComputeEVGradFn<TKey, TValue, Mean>, batch_size,
                              this->dimension_, this->max_norm_, stream);
    } else if (this->combiner_ == "sum") {
      this->lookuper_.compute(ComputeEVGradFn<TKey, TValue, Sum>, batch_size,
                              this->dimension_, this->max_norm_, stream);
    } else {
      this->lookuper_.compute(ComputeEVGradFn<TKey, TValue, Sqrtn>, batch_size,
                              this->dimension_, this->max_norm_, stream);
    }
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype) \
  REGISTER_KERNEL_BUILDER(                                 \
      Name("MultiKvResourceGatherGrad")                    \
          .Device(DEVICE_GPU)                              \
          .TypeConstraint<key_type_tf>("Tkeys")            \
          .TypeConstraint<dtype>("dtype"),                 \
      MultiKvResourceGatherBackWardOp<key_type_tf, key_type, dtype>)

REGISTER_GPU_KERNELS(int64, int64_t, float);
REGISTER_GPU_KERNELS(int32, int32_t, float);
#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow

#endif // GOOGLE_CUDA
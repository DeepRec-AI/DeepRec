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

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"
#include "tensorflow/core/kernels/group_lookup_backward_base_ops.cu.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename TFKey, typename TKey, typename TValue>
class GroupVariableLookupBackwardOp
    : public GroupLookupBackWardBaseOp<TKey, TValue> {
 public:
  explicit GroupVariableLookupBackwardOp(OpKernelConstruction* c)
      : GroupLookupBackWardBaseOp<TKey, TValue>(c) {}

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    int batch_size = -1;

    Allocator* gpu_allocator =
        ctx->device()->GetAllocator(AllocatorAttributes());
    GroupEmbeddingLookupBackWard<TKey, TValue> lookuper(this->dimension_, this->num_lookups_,
                                          this->max_norm_, gpu_allocator);
    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor grads_tensor = ctx->input(i);
      const Tensor emb_variables_tensor = ctx->input(this->num_lookups_ + i);
      const Tensor sp_values_tensor = ctx->input(2 * this->num_lookups_ + i);
      const Tensor sp_values_offset_tensor =
          ctx->input(3 * this->num_lookups_ + i);
      const int64_t nnz = sp_values_tensor.NumElements();

      Tensor* grads_sp_values_tensor;
      TensorShape grads_sp_values_tensor_shape =
          TensorShape(std::vector<int64>({nnz, this->dimension_}));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, grads_sp_values_tensor_shape,
                                               &grads_sp_values_tensor));
      auto* grads_sp_values = grads_sp_values_tensor->flat<TValue>().data();
      cudaMemsetAsync(grads_sp_values, 0,
                      sizeof(TValue) * nnz * this->dimension_, stream);

      if (i == 0) {
        batch_size = sp_values_offset_tensor.shape().dim_size(0);
      }

      GroupEmbeddingBackWardArgs<TKey, TValue> args(
          const_cast<TValue*>(grads_tensor.flat<TValue>().data()),
          const_cast<TKey*>(reinterpret_cast<const TKey*>(
              sp_values_tensor.flat<TFKey>().data())),
          const_cast<TValue*>(emb_variables_tensor.flat<TValue>().data()),
          grads_sp_values,
          const_cast<int*>(sp_values_offset_tensor.flat<int>().data()), nnz);
      lookuper.set(args);
    }

    if (this->combiner_ == "mean") {
      this->template compute<false, Mean>(lookuper, batch_size, stream);
    } else if (this->combiner_ == "sum") {
      this->template compute<false, Sum>(lookuper, batch_size, stream);
    } else {
      this->template compute<false, Sqrtn>(lookuper, batch_size, stream);
    }
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype) \
  REGISTER_KERNEL_BUILDER(                                 \
      Name("GroupVariableLookupGrad")                      \
          .Device(DEVICE_GPU)                              \
          .TypeConstraint<key_type_tf>("Tkeys")            \
          .TypeConstraint<dtype>("dtype"),                 \
      GroupVariableLookupBackwardOp<key_type_tf, key_type, dtype>)

REGISTER_GPU_KERNELS(int64, int64_t, float);
REGISTER_GPU_KERNELS(int32, int32_t, float);
#undef REGISTER_GPU_KERNELS

template <typename TFKey, typename TKey, typename TValue>
class GroupEmbeddingVariableLookupBackwardOp
    : public GroupLookupBackWardBaseOp<TKey, TValue> {
 public:
  explicit GroupEmbeddingVariableLookupBackwardOp(OpKernelConstruction* c)
      : GroupLookupBackWardBaseOp<TKey, TValue>(c) {}

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    int batch_size = -1;

    Allocator* gpu_allocator =
        ctx->device()->GetAllocator(AllocatorAttributes());
    GroupEmbeddingLookupBackWard<TKey, TValue> lookuper(this->dimension_, this->num_lookups_,
                                          this->max_norm_, gpu_allocator);
    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor grads_tensor = ctx->input(i);
      EmbeddingVar<TFKey, TValue>* ev = nullptr;
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, this->num_lookups_ + i),
                              &ev));
      core::ScopedUnref unref_me(ev);
      const Tensor sp_values_tensor = ctx->input(2 * this->num_lookups_ + i);
      const Tensor sp_values_offset_tensor =
          ctx->input(3 * this->num_lookups_ + i);

      const int64_t nnz = sp_values_tensor.NumElements();

      Tensor* grads_sp_values_tensor;
      TensorShape grads_sp_values_tensor_shape =
          TensorShape(std::vector<int64>({nnz, this->dimension_}));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, grads_sp_values_tensor_shape,
                                               &grads_sp_values_tensor));
      auto* grads_sp_values = grads_sp_values_tensor->flat<TValue>().data();
      cudaMemsetAsync(grads_sp_values, 0,
                      sizeof(TValue) * nnz * this->dimension_, stream);

      if (i == 0) {
        batch_size = sp_values_offset_tensor.shape().dim_size(0);
      }

      GroupEmbeddingBackWardArgs<TKey, TValue> args(
          const_cast<TValue*>(grads_tensor.flat<TValue>().data()),
          const_cast<TKey*>(reinterpret_cast<const TKey*>(
              sp_values_tensor.flat<TFKey>().data())),
          nullptr /*fake*/, grads_sp_values,
          const_cast<int*>(sp_values_offset_tensor.flat<int>().data()), nnz);
      lookuper.set(args);
    }

    if (this->combiner_ == "mean") {
      this->template compute<true, Mean>(lookuper, batch_size, stream);
    } else if (this->combiner_ == "sum") {
      this->template compute<true, Sum>(lookuper, batch_size, stream);
    } else {
      this->template compute<true, Sqrtn>(lookuper, batch_size, stream);
    }
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype) \
  REGISTER_KERNEL_BUILDER(                                 \
      Name("GroupEmbeddingVariableLookupGrad")             \
          .Device(DEVICE_GPU)                              \
          .TypeConstraint<key_type_tf>("Tkeys")            \
          .TypeConstraint<dtype>("dtype"),                 \
      GroupEmbeddingVariableLookupBackwardOp<key_type_tf, key_type, dtype>)

REGISTER_GPU_KERNELS(int64, int64_t, float);
REGISTER_GPU_KERNELS(int32, int32_t, float);
#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
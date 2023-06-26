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
#endif

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/embedding_var_context.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif //GOOGLE_CUDA

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#if GOOGLE_CUDA
using se::DeviceMemoryBase;
using se::Stream;
#endif //GOOGLE_CUDA

template <typename TKey, typename TValue>
class KvResourceLookupResourceOp : public OpKernel {
 public:
  explicit KvResourceLookupResourceOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {1}, &output));
    auto output_scalar = output->scalar<int64>();
    output_scalar() = (int64)ev;
  }
};

#define REGISTER_KV_LOOKUP_RESOURCE(dev, ktype, vtype)                 \
  REGISTER_KERNEL_BUILDER(Name("KvResourceLookupResource")             \
                          .Device(DEVICE_##dev)                        \
                          .HostMemory("output")                        \
                          .TypeConstraint<ktype>("Tkeys")              \
                          .TypeConstraint<vtype>("dtype"),             \
                          KvResourceLookupResourceOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)                                \
  REGISTER_KV_LOOKUP_RESOURCE(dev, int32, type)                        \
  REGISTER_KV_LOOKUP_RESOURCE(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KV_LOOKUP_RESOURCE

template <typename Device, typename TKey, typename TValue>
class KvResourceLookupIDOp : public OpKernel {
 public:
  explicit KvResourceLookupIDOp(OpKernelConstruction* c) : OpKernel(c) {
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = c->input(1);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    if (N > 0) {
      auto out_flat = out->flat<int64>();
      int64* out_base = &out_flat(0);

      auto indices_flat = indices.flat<TKey>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      EmbeddingVarContext<Device> ev_ctx(c);
      ev->GetOrCreateKey(ev_ctx, indices,
                         reinterpret_cast<ValuePtr<TValue>**>(out_base),
                         indices_size);
    }
  }
};

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceLookupID")         \
                              .Device(DEVICE_##dev)               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceLookupIDOp<CPUDevice, ktype, vtype>)
#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceLookupID")         \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("indices")              \
                              .HostMemory("pointer")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceLookupIDOp<GPUDevice, ktype, vtype>)
#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

template <typename Device, typename TKey, typename TValue>
class KvResourceCollectEmbeddingOp : public OpKernel {
 public:
  explicit KvResourceCollectEmbeddingOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = c->input(1);
    const Tensor& pointer = c->input(2);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({ev->ValueLen()});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      auto pointer_flat = pointer.flat<int64>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      OP_REQUIRES(c, ev->ValueLen() == slice_elems,
          errors::InvalidArgument(
              "ev's value_len should same with output's dimension(1)",
              std::to_string(slice_elems), std::to_string(ev->ValueLen())));
      OP_REQUIRES(c, !ev->IsMultiLevel() ||
          (ev->IsMultiLevel() && ev->CacheSize() >= N),
          errors::InvalidArgument(
              "MultiLevel EV's Cache size ", ev->CacheSize(),
              " should large than IDs in batch ", N));
      const size_t slice_bytes = slice_elems * sizeof(TValue);
      EmbeddingVarContext<Device> ev_ctx(c);
      ev->GatherEmbeddings(ev_ctx, indices,
                          (ValuePtr<TValue>**)pointer.data(),
                          out_base, N);
    }
  }
};

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceCollectEmbedding") \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("pointer")              \
                              .HostMemory("default_value")        \
                              .HostMemory("output")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceCollectEmbeddingOp<CPUDevice, ktype, vtype>)

#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceCollectEmbedding") \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("indices")              \
                              .HostMemory("pointer")              \
                              .HostMemory("default_value")        \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceCollectEmbeddingOp<GPUDevice, ktype, vtype>)

#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif //GOOGLE_CUDA

template <typename TKey, typename TValue, bool has_counts>
class KvResourceGatherOp : public OpKernel {
 public:
  explicit KvResourceGatherOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = c->input(1);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({ev->ValueLen()});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      const int64 slice_elems = out_flat.dimension(1);
      OP_REQUIRES(c, ev->ValueLen() == slice_elems,
          errors::InvalidArgument(
              "ev's value_len should same with output's dimension(1)",
              std::to_string(slice_elems), std::to_string(ev->ValueLen())));
      OP_REQUIRES(c, !ev->IsMultiLevel() ||
          (ev->IsMultiLevel() && ev->CacheSize() >= N),
          errors::InvalidArgument(
              "MultiLevel EV's Cache size ", ev->CacheSize(),
              " should large than IDs in batch ", N));

      EmbeddingVarContext<CPUDevice> ev_ctx(c);
      if (is_use_default_value_tensor_) {
        ev->GetEmbeddings(ev_ctx, (TKey*)indices.data(), out_base, N,
                          reinterpret_cast<TValue*>(c->input(2).data()));
      } else {
        ev->GetEmbeddings(ev_ctx, (TKey*)indices.data(), out_base, N);
        if (has_counts) {
          const Tensor& indices_counts = c->input(2);
          ev->UpdateCache(indices, indices_counts, true);
        } else {
          ev->UpdateCache(indices, true);
        }
      }
    }
  }

  private:
    bool is_use_default_value_tensor_;
};

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGather")                \
                              .Device(DEVICE_##dev)               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOp<ktype, vtype, false>)

#define REGISTER_KERNELS_ALL_INDICES(type)                        \
  REGISTER_KERNELS(CPU, int32, type);                             \
  REGISTER_KERNELS(CPU, int64, type)

TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDICES)
#undef REGISTER_KERNELS_ALL_INDICES
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGatherV1")              \
                              .Device(DEVICE_##dev)               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOp<ktype, vtype, true>)

#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
template <typename Device, typename TKey, typename TValue, bool has_counts>
class KvResourceGatherGPUOp : public OpKernel {
 public:
  explicit KvResourceGatherGPUOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
    bool is_inference;
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference));
    if (!is_inference) {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, const TKey* key,
                      TValue* val, TValue* default_v, int32 default_v_num,
                      bool is_use_default_value_tensor,
                      size_t n, const Eigen::GpuDevice& device) {
        ev->LookupOrCreate(key, val, default_v, default_v_num,
            is_use_default_value_tensor, n, device);
      };
    } else {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, const TKey* key,
                      TValue* val, TValue* default_v, int32 default_v_num,
                      bool is_use_default_value_tensor,
                      size_t n, const Eigen::GpuDevice& device) {
        ev->Lookup(key, val, default_v, default_v_num,
            is_use_default_value_tensor, n, device);
      };
    }
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = c->input(1);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({ev->ValueLen()});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      TValue* default_v = nullptr;
      if (is_use_default_value_tensor_) {
        default_v = (TValue*)c->input(2).data();
      } else {
        default_v = ev->GetDefaultValuePtr();
      }
      OP_REQUIRES(c, ev->ValueLen() == slice_elems,
          errors::InvalidArgument(
              "ev's value_len should same with output's dimension(1)",
              std::to_string(slice_elems), std::to_string(ev->ValueLen())));
      OP_REQUIRES(c, !ev->IsMultiLevel() ||
          (ev->IsMultiLevel() && ev->CacheSize() >= N),
          errors::InvalidArgument(
              "MultiLevel EV's Cache size ", ev->CacheSize(),
              " should large than IDs in batch ", N));
      const size_t slice_bytes = slice_elems * sizeof(TValue);
      if (ev->IsSingleHbm()) {
        const TKey* key_base = &indices_flat(0);
        const Device& device = c->eigen_device<Device>();
        if (is_use_default_value_tensor_) {
          Tensor default_values(c->input(2));
          auto default_value_num = default_values.NumElements() / ev->ValueLen();
          auto default_values_matrix = default_values.shaped<TValue, 2>(
              {default_value_num, ev->ValueLen()});
          TValue* default_v_base = &default_values_matrix(0, 0);
          lookup_fn_(ev, key_base, out_base, default_v_base,
              default_value_num, is_use_default_value_tensor_,
              indices_size, device);
        } else {
          lookup_fn_(ev, key_base, out_base, ev->GetDefaultValuePtr(),
              ev->GetDefaultValueDim(), is_use_default_value_tensor_,
              indices_size, device);
        }
      } else {
        Tensor indices_host(indices.dtype(), indices.shape());
        //Copy ids from GPU to CPU for CPU Lookup.
        auto stream = c->op_device_context()->stream();
        auto event_mgr = c->device()->tensorflow_gpu_device_info()->event_mgr;
        se::DeviceMemoryBase gpu_src(
            const_cast<TKey*>(&indices_flat(0)), N * sizeof(TKey));
        stream->ThenMemcpy(indices_host.data(), gpu_src, N * sizeof(TKey));
        SyncWithEventMgr(stream, event_mgr);

        EmbeddingVarContext<GPUDevice> ev_ctx(c);
        ev->GetEmbeddings(ev_ctx, (TKey*)indices_host.data(),
                          out_base, N);
        if (has_counts) {
          const Tensor& indices_counts = c->input(2);
          ev->UpdateCache(indices_host, indices_counts, true);
        } else {
          ev->UpdateCache(indices_host, true);
        }
      }
    }
  }

  private:
    bool is_use_default_value_tensor_;
    std::function<void(EmbeddingVar<TKey, TValue>* ev, const TKey* key,
                      TValue* val, TValue* default_v, int32 default_v_num,
                      bool is_use_default_value_tensor,
                      size_t n, const Eigen::GpuDevice& device)> lookup_fn_;
};

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGather")                \
                              .Device(DEVICE_##dev)               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherGPUOp<GPUDevice, ktype, vtype, false>)

#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU);
#undef REGISTER_KERNELS_GPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGatherV1")              \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("counts")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherGPUOp<GPUDevice, ktype, vtype, true>)

#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

template <typename TKey, typename TValue>
class EVGetFrequencyOp : public OpKernel {
 public:
  explicit EVGetFrequencyOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = ctx->input(1);
    auto indices_flat = indices.flat<TKey>();

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {indices.NumElements()}, &output));
    for (int i = 0; i < indices.NumElements(); ++i) {
      int64 f = ev->GetFreq(indices_flat(i));
      output->flat<int64>()(i) = f;
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVGetFrequency")                \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("Tvalues"),  \
                          EVGetFrequencyOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class EVGetVersionOp : public OpKernel {
 public:
  explicit EVGetVersionOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = ctx->input(1);
    auto indices_flat = indices.flat<TKey>();

    Tensor* output;
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, {indices.NumElements()}, &output));
    for (int i = 0; i < indices.NumElements(); ++i) {
      int64 v = ev->GetVersion(indices_flat(i));
      output->flat<int64>()(i) = v;
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVGetVersion")                  \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("Tvalues"),  \
                          EVGetVersionOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceLookupTierOp : public OpKernel {
 public:
  explicit KvResourceLookupTierOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = ctx->input(1);
    auto indices_flat = indices.flat<TKey>();

    Tensor* output;
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, {indices.NumElements()}, &output));
    for (int i = 0; i < indices.NumElements(); ++i) {
      int v = ev->storage()->LookupTier(indices_flat(i));
      output->flat<int>()(i) = v;
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("KvResourceLookupTier")          \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          KvResourceLookupTierOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("KvResourceLookupTier")          \
                            .Device(DEVICE_GPU)                 \
                            .HostMemory("ids")                  \
                            .HostMemory("output")               \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          KvResourceLookupTierOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

}  // namespace tensorflow


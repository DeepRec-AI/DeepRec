/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
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

namespace tensorflow {

namespace {
const int64 kEmbeddingVarUseDB = -214;
const int64 kInitializableEmbeddingVarUseDB = -215;
const char* kInferenceMode = "INFERENCE_MODE";
}

#define REGISTER_KV_VAR_HANDLE(ktype, vtype)                           \
  REGISTER_KERNEL_BUILDER(Name("KvVarHandleOp")                        \
                          .Device(DEVICE_CPU)                          \
                          .TypeConstraint<ktype>("Tkeys")              \
                          .TypeConstraint<vtype>("dtype"),             \
                          ResourceHandleOp<EmbeddingVar<ktype, vtype>>);
#define REGISTER_KERNELS_ALL_INDEX(type)                               \
  REGISTER_KV_VAR_HANDLE(int32, type)                                  \
  REGISTER_KV_VAR_HANDLE(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KV_VAR_HANDLE

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#define REGISTER_KV_VAR_HANDLE(ktype, vtype)                           \
  REGISTER_KERNEL_BUILDER(Name("KvVarHandleOp")                        \
                          .Device(DEVICE_GPU)                          \
                          .TypeConstraint<ktype>("Tkeys")              \
                          .TypeConstraint<vtype>("dtype"),             \
                          ResourceHandleOp<EmbeddingVar<ktype, vtype>>);
REGISTER_KV_VAR_HANDLE(int32, float)
REGISTER_KV_VAR_HANDLE(int64, float)
#undef REGISTER_KV_VAR_HANDLE
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

template <typename T, typename TKey, typename TValue>
class KvVariableShapeOp : public OpKernel {
 public:
  explicit KvVariableShapeOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    TensorShape shape({ev->Size(), ev->ValueLen()});
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {shape.dims()}, &output));
    for (int i = 0; i < shape.dims(); ++i) {
      output->flat<T>()(i) = shape.dim_size(i);
    }
  }
};

#define REGISTER_KERNELS(type, ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("KvVariableShape").Device(DEVICE_CPU)                      \
                             .TypeConstraint<type>("out_type")        \
                             .TypeConstraint<ktype>("Tkeys")          \
                             .TypeConstraint<vtype>("dtype"),         \
                             KvVariableShapeOp<type, ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                              \
  REGISTER_KERNELS(int32, int32, type)                                \
  REGISTER_KERNELS(int32, int64, type)                                \
  REGISTER_KERNELS(int64, int32, type)                                \
  REGISTER_KERNELS(int64, int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

class DestroyKvResourceOp : public OpKernel {
 public:
  explicit DestroyKvResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
        ctx->GetAttr("ignore_lookup_error", &ignore_lookup_error_));
  }

  void Compute(OpKernelContext* ctx) override {
    const ResourceHandle& p = HandleFromInput(ctx, 0);
    Status status = DeleteResource(ctx, p);
    if (ignore_lookup_error_ && errors::IsNotFound(status)) {
      return;
    }
    OP_REQUIRES_OK(ctx, status);
  }

 private:
  bool ignore_lookup_error_;
};

REGISTER_KERNEL_BUILDER(Name("DestroyKvResourceOp").Device(DEVICE_CPU),
                        DestroyKvResourceOp);

template <typename TKey, typename TValue>
class InitializeKvVariableOp : public OpKernel {
 public:
  explicit InitializeKvVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));
    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
          &false_positive_probability_));
    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold",
          &l2_weight_threshold_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim", &default_value_dim_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_no_permission",
          &default_value_no_permission_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));
    OP_REQUIRES_OK(c, c->GetAttr("record_freq", &record_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("record_version", &record_version_));

    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));
    OP_REQUIRES_OK(c, c->GetAttr("storage_size", &storage_size_));

    if (filter_freq_ < 0) {
      LOG(INFO) << "filter_freq < 0 is invalid, feature filter is disabled.";
      filter_freq_ = 0;
    }

    if ((filter_freq_ != 0 && max_element_size_ == 0)
         || steps_to_live_ != 0 || record_freq_
         || record_version_ || storage_type > 5) {
      if (block_num_ > 1 || (filter_freq_ != 0 && storage_type <= 5)) {
        layout_ = "normal";
      } else {
        if (storage_type == embedding::HBM_DRAM ||
            storage_type == embedding::HBM_DRAM_SSDHASH) {
          layout_ = "normal_contiguous_gpu";
        } else {
          layout_ = "normal_contiguous";
        }
      }
    } else {
      layout_ = "light";
    }

    CHECK(block_num_ == 1 || layout_ != "normal_contiguous");

    if (steps_to_live_ == kEmbeddingVarUseDB ||
        steps_to_live_ == kInitializableEmbeddingVarUseDB) {
      LOG(INFO) << "hashmap use db";
      //use_db_ = true;
    } else {
      OP_REQUIRES(c, steps_to_live_ >= 0,
          errors::InvalidArgument(
            "steps_to_live must >= 0, ", std::to_string(steps_to_live_)));
    }
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    if (embedding::StorageType::LEVELDB == storage_type_) {
      ht_type_ = "leveldb_kv";
      if (layout_ != "normal_contiguous")
        LOG(WARNING)
          << "layout must be NORAML_CONTIGUOUS when storage type is LEVELDB";
      layout_ = "normal_contiguous";
    }

    if (embedding::StorageType::PMEM_LIBPMEM == storage_type_ ||
        embedding::StorageType::PMEM_MEMKIND == storage_type_){
      if (layout_ != "normal_contiguous"){
        LOG(WARNING)
          << "layout must be NORAML_CONTIGUOUS"
          << " when storage type is PMEM_LIBPMEM or PMEM_MEMKIND";
      }
      layout_ = "normal_contiguous";
    }
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& default_values = context->input(2);

    OP_REQUIRES(context, dtype_ == default_values.dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 0);
    ResourceHandle handle_primary = HandleFromInput(context, 1);
    std::string opname = handle_self.name();

    EmbeddingVar<TKey, TValue>* ev = nullptr;

    if (handle_self.name() == handle_primary.name() &&
        handle_self.container() == handle_primary.container()) {

      OP_REQUIRES_OK(context,
          LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, context,
             handle_self](EmbeddingVar<TKey, TValue>** ptr) {
              Allocator* gpu_allocator =
                  context->device()->GetAllocator(AllocatorAttributes());
              auto storage_manager =
                  new embedding::StorageManager<TKey, TValue>(
                    handle_self.name(),
                    embedding::StorageConfig(
                      storage_type_, storage_path_, storage_size_, layout_),
                    gpu_allocator);
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                         storage_manager,
                         EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                             emb_index_, block_num_, slot_num_,
                             opname + "-primary", steps_to_live_,
                             filter_freq_, max_freq_,
                             l2_weight_threshold_, layout_,
                             max_element_size_, false_positive_probability_,
                             counter_type_, default_value_dim_,
                             default_value_no_permission_,
                             record_freq_, record_version_),
                         gpu_allocator);
            return Status::OK();
            }));
      ev->Init(default_values, default_value_dim_);
    } else {
      EmbeddingVar<TKey, TValue>* primary_variable = nullptr;
      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname,
            handle_primary, context](EmbeddingVar<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             Allocator* gpu_allocator = context->device()->GetAllocator(AllocatorAttributes());
             auto storage_manager =
               new embedding::StorageManager<TKey, TValue>(
                 handle_primary.name(), embedding::StorageConfig(storage_type_,
                     storage_path_, storage_size_, layout_), gpu_allocator);
             *ptr = new EmbeddingVar<TKey, TValue>(handle_primary.name(),
                        storage_manager,
                        EmbeddingConfig(
                          primary_emb_index + block_num_ * primary_slot_index,
                          primary_emb_index,
                          block_num_, slot_num_, opname + "-primary",
                          steps_to_live_, filter_freq_, max_freq_,
                          l2_weight_threshold_, layout_,
                          max_element_size_, false_positive_probability_,
                          counter_type_, 0, record_freq_, record_version_),
                        gpu_allocator);
            // default_values is slot value, should not to initialize primary value
            return Status::OK();
           }));

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, primary_variable,
             handle_self, context](EmbeddingVar<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                  primary_variable->storage_manager(),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                                  emb_index_,
                                  block_num_, slot_num_, opname,
                                  steps_to_live_, filter_freq_,
                                  max_freq_, l2_weight_threshold_,
                                  layout_, max_element_size_,
                                  false_positive_probability_,
                                  counter_type_, default_value_dim_,
                                  default_value_no_permission_,
                                  record_freq_, record_version_),
                  primary_variable->GetAllocator());
             return (*ptr)->Init(default_values, default_value_dim_);
            }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);
    if (steps_to_live_ != kEmbeddingVarUseDB) {
      ev->SetInitialized();
    }
  }

 private:
  DataType dtype_;
  DataType counter_type_;
  TensorShape shape_;
  int64 steps_to_live_;
  int64 emb_index_;
  int64 block_num_;
  int64 slot_index_;
  int64 slot_num_;
  std::string ht_type_;
  int64 ht_partition_num_;
  int64 filter_freq_;
  int64 max_freq_;
  float l2_weight_threshold_;
  std::string layout_;
  int64 max_element_size_;
  float false_positive_probability_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  std::vector<int64> storage_size_;
  int64 default_value_dim_;
  float default_value_no_permission_;
  bool record_freq_;
  bool record_version_;
};

#define REGISTER_KERNELS(ktype, vtype)                               \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableOp")             \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<ktype>("Tkeys")        \
                              .TypeConstraint<vtype>("dtype"),       \
                          InitializeKvVariableOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                             \
  REGISTER_KERNELS(int32, type)                                      \
  REGISTER_KERNELS(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#define REGISTER_KERNELS(ktype, vtype)                               \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableOp")             \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("resource_self")           \
                              .HostMemory("resource_primary")        \
                              .HostMemory("value")                   \
                              .HostMemory("empty_key")               \
                              .TypeConstraint<ktype>("Tkeys")        \
                              .TypeConstraint<vtype>("dtype"),       \
                          InitializeKvVariableOp<ktype, vtype>);

#define REGISTER_GPU_KERNELS(T)        \
  REGISTER_KERNELS(int32, T);          \
  REGISTER_KERNELS(int64, T);

TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_double(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

template <typename TKey, typename TValue>
class KvResourceIsInitializedOp : public OpKernel {
 public:
  explicit KvResourceIsInitializedOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    bool found;
    if (LookupResource<EmbeddingVar<TKey, TValue>>(
          ctx, HandleFromInput(ctx, 0), &ev).ok()) {
      found = ev->IsInitialized();
      ev->Unref();
    } else {
      found = false;
    }

    output->flat<bool>()(0) = found;
  }
};

#if GOOGLE_CUDA
#if TENSORFLOW_USE_GPU_EV
template <typename TKey, typename TValue>
class KvResourceIsInitializedGPUOp : public OpKernel {
 public:
  explicit KvResourceIsInitializedGPUOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
    EmbeddingVarGPU<TKey, TValue>* ev = nullptr;
    bool found;
    if (LookupResource<EmbeddingVarGPU<TKey, TValue>>(
          ctx, HandleFromInput(ctx, 0), &ev).ok()) {
      found = ev->IsInitialized();
      ev->Unref();
    } else {
      found = false;
    }

    output->flat<bool>()(0) = found;
  }
};
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

#define REGISTER_KERNELS(ktype, vtype)                             \
  REGISTER_KERNEL_BUILDER(Name("KvVarIsInitializedOp")             \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .TypeConstraint<vtype>("dtype")          \
                          .Device(DEVICE_CPU),                     \
                          KvResourceIsInitializedOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                           \
  REGISTER_KERNELS(int32, type)                                    \
  REGISTER_KERNELS(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#define REGISTER_KERNELS(ktype, vtype)                             \
  REGISTER_KERNEL_BUILDER(Name("KvVarIsInitializedOp")             \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .Device(DEVICE_GPU)                      \
                          .HostMemory("is_initialized"),           \
                          KvResourceIsInitializedOp<ktype, vtype>);
#else
#define REGISTER_KERNELS(ktype, vtype)                             \
  REGISTER_KERNEL_BUILDER(Name("KvVarIsInitializedOp")             \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .Device(DEVICE_GPU)                      \
                          .HostMemory("is_initialized"),           \
                          KvResourceIsInitializedGPUOp<ktype, vtype>);
#endif  // TENSORFLOW_USE_GPU_EV
REGISTER_KERNELS(int32, float)
REGISTER_KERNELS(int64, float)
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

template <typename TKey, typename TValue>
class KvResourceInitCacheStrategyOp : public OpKernel {
 public:
  explicit KvResourceInitCacheStrategyOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("cache_strategy", &cache_strategy_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    ev->InitCache(static_cast<embedding::CacheStrategy>(cache_strategy_));
  }

 private:
  int cache_strategy_;
};

#define REGISTER_KERNELS(ktype, vtype)                             \
  REGISTER_KERNEL_BUILDER(Name("KvResourceInitCacheStrategyOp")             \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .TypeConstraint<vtype>("dtype")          \
                          .Device(DEVICE_CPU),                     \
                          KvResourceInitCacheStrategyOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                           \
  REGISTER_KERNELS(int32, type)                                    \
  REGISTER_KERNELS(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#define REGISTER_KERNELS(ktype, vtype)                             \
  REGISTER_KERNEL_BUILDER(Name("KvResourceInitCacheStrategyOp")             \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .Device(DEVICE_GPU),                     \
                          KvResourceInitCacheStrategyOp<ktype, vtype>);
REGISTER_KERNELS(int32, float)
REGISTER_KERNELS(int64, float)
#undef REGISTER_KERNELS
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

template <typename TKey, typename TValue>
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
      auto do_work = [this, indices_flat,
           out_base, ev] (int64 start, int64 limit) {
        for (int64 i = start; i < limit; ++i) {
          ValuePtr<TValue>* value_ptr;
          bool is_filter = false;
          ev->LookupOrCreateKey(indices_flat(i), &value_ptr, &is_filter, false);
          *(out_base + i) = (int64)value_ptr;
        }
      };

      auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads->num_threads, worker_threads->workers, indices_size,
          100, do_work);

      if (ev->IsMultiLevel()) {
        ev->storage_manager()->Schedule([ev, indices]() {
          embedding::BatchCache<TKey>* cache = ev->Cache();
          if (cache) {
            cache->add_to_rank(indices);
          }
        });
      }
    }
  }
};

#define REGISTER_LOOKUP_FULL(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceLookupID")         \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceLookupIDOp<ktype, vtype>)

#define REGISTER_LOOKUP_ALL_INDICES(dev, type)                    \
  REGISTER_LOOKUP_FULL(dev, int32, type);                         \
  REGISTER_LOOKUP_FULL(dev, int64, type)

#define REGISTER_LOOKUP_CPU(type) REGISTER_LOOKUP_ALL_INDICES(CPU, type)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_LOOKUP_CPU)

#undef REGISTER_LOOKUP_CPU
#undef REGISTER_LOOKUP_ALL_INDICES
#undef REGISTER_LOOKUP_FULL

template <typename TKey, typename TValue>
class KvResourceGatherOp : public OpKernel {
 public:
  explicit KvResourceGatherOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("is_inference", &is_inference_));
    bool is_inference;
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference));
    is_inference_ |= is_inference;
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * index;
      };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim) ;
      };
    }
    if (c->num_inputs() == 4) {
      get_count_fn_ = [](const int32* count, int64 index) {
        return count[index];
      };
    } else {
      get_count_fn_ = [](const int32* count, int64 index) {
        return 1;
      };
    }
    if (!is_inference_) {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key,
                      TValue* val, TValue* default_v, int count) {
        ev->LookupOrCreate(key, val, default_v, count);
        return Status::OK();
      };
    } else {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key,
                      TValue* val, TValue* default_v, int count) {
        Status s = ev->Lookup(key, val, default_v);
        return s;
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

    int32* counts = nullptr;
    if (c->num_inputs() == 4)
      counts = (int32*)c->input(3).data();

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
      auto do_work = [this, indices_flat,
           out_base, slice_elems, c, default_v, ev, counts] (
               int64 start, int64 limit) {
        for (int64 i = start; i < limit; ++i) {
          TValue* default_v_ptr = get_default_v_fn_(
              default_v, indices_flat(i), i, ev->GetDefaultValueDim(),
              ev->ValueLen());
          int32 count = get_count_fn_(counts, i);
          OP_REQUIRES_OK(c, lookup_fn_(ev, indices_flat(i),
              out_base + i * slice_elems, default_v_ptr, count));
        }
      };
      auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads->num_threads,
            worker_threads->workers, indices_size,
            slice_bytes, do_work);

      if (ev->IsMultiLevel()) {
        embedding::BatchCache<TKey>* cache = ev->Cache();
        ev->storage_manager()->Schedule([ev, indices]() {
          embedding::BatchCache<TKey>* cache = ev->Cache();
          cache->add_to_rank(indices);
        });
      }
    }
  }

  private:
    bool is_use_default_value_tensor_;
    bool is_inference_;
    std::function<
      TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
    std::function<int32(int32*, int64)> get_count_fn_;
    std::function<Status(EmbeddingVar<TKey, TValue>* ev,
      TKey key, TValue* val, TValue* default_v, int count)> lookup_fn_;
};

#define REGISTER_GATHER_FULL(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGather")                \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("default_value")        \
                              .HostMemory("output")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOp<ktype, vtype>)

#define REGISTER_GATHER_ALL_INDICES(type)                         \
  REGISTER_GATHER_FULL(CPU, int32, type);                         \
  REGISTER_GATHER_FULL(CPU, int64, type)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GATHER_ALL_INDICES)
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

template <typename TKey, typename TValue>
class KvResourceCollectEmbeddingOp : public OpKernel {
 public:
  explicit KvResourceCollectEmbeddingOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * index;
      };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim) ;
      };
    }
    if (c->num_inputs() == 5) {
      get_count_fn_ = [](const int32* count, int64 index) {
        return count[index];
      };
    } else {
      get_count_fn_ = [](const int32* count, int64 index) {
        return 1;
      };
    }
    lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key,
                    TValue* val, TValue* default_v, int count) {
      if (key) {
        TValue* mem_val = ev->LookupOrCreateEmb((ValuePtr<TValue>*)key, default_v);
        memcpy(val, mem_val, sizeof(TValue) * ev->ValueLen());
      } else {
        memcpy(val, default_v, sizeof(TValue) * ev->ValueLen());
      }
      return Status::OK();
    };
  }

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

    int32* counts = nullptr;
    if (c->num_inputs() == 5)
      counts = (int32*)c->input(4).data();

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      auto pointer_flat = pointer.flat<int64>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      TValue* default_v = nullptr;
      if (is_use_default_value_tensor_) {
        default_v = (TValue*)c->input(3).data();
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
      auto do_work = [this, indices_flat, pointer_flat,
           out_base, slice_elems, c, default_v, ev, counts] (
               int64 start, int64 limit) {
        for (int64 i = start; i < limit; ++i) {
          TValue* default_v_ptr = get_default_v_fn_(
              default_v, indices_flat(i), i, ev->GetDefaultValueDim(),
              ev->ValueLen());
          int32 count = get_count_fn_(counts, i);
          OP_REQUIRES_OK(c, lookup_fn_(ev, pointer_flat(i),
              out_base + i * slice_elems, default_v_ptr, count));
        }
      };
      auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads->num_threads,
            worker_threads->workers, indices_size,
            slice_bytes, do_work);
    }
  }

  private:
    bool is_use_default_value_tensor_;
    std::function<
      TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
    std::function<int32(int32*, int64)> get_count_fn_;
    std::function<Status(EmbeddingVar<TKey, TValue>* ev,
      TKey key, TValue* val, TValue* default_v, int count)> lookup_fn_;
};

#define REGISTER_GATHER_FULL(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceCollectEmbedding") \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("pointer")              \
                              .HostMemory("default_value")        \
                              .HostMemory("output")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceCollectEmbeddingOp<ktype, vtype>)

#define REGISTER_GATHER_ALL_INDICES(type)                         \
  REGISTER_GATHER_FULL(CPU, int32, type);                         \
  REGISTER_GATHER_FULL(CPU, int64, type)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GATHER_ALL_INDICES)
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
template <typename TKey, typename TValue>
class KvResourceGatherGPUOp : public OpKernel {
 public:
  explicit KvResourceGatherGPUOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * index;
      };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim) ;
      };
    }
    if (c->num_inputs() == 4) {
      get_count_fn_ = [](const int32* count, int64 index) {
        return count[index];
      };
    } else {
      get_count_fn_ = [](const int32* count, int64 index) {
        return 1;
      };
    }
    hash_map_.max_load_factor(0.8);
    hash_map_.set_empty_key_and_value(-1, -1);
    hash_map_.set_counternum(16);
    hash_map_.set_deleted_key(-2);
  }

  ~KvResourceGatherGPUOp() {
    delete[] occupy_flag_;
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

    int32* counts = nullptr;
    if (c->num_inputs() == 4)
      counts = (int32*)c->input(3).data();

    int64 num_threads = c->device()
                         ->tensorflow_cpu_worker_threads()
                         ->num_threads;
    if (occupy_flag_ == nullptr) {
      mutex_lock l(m_init_occupy_flag_);
      //double check
      if (occupy_flag_ == nullptr) {
        occupy_flag_ = new bool[num_threads];
        memset(occupy_flag_, 0, sizeof(bool) * num_threads);
      }
    }

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
      if (ev->IsUseHbm()) {
        TValue** memcpy_address = new TValue*[indices_size];
        auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
        std::vector<std::list<int64>> init_cursor_list(
                                          worker_threads->num_threads + 1);
        std::vector<std::list<int64>> copyback_cursor_list(
                                          worker_threads->num_threads + 1);
        int64 main_thread_id = Env::Default()->GetCurrentThreadId();
        auto do_work = [this, indices_flat,
            out_base, slice_elems, c, ev,
            memcpy_address, &init_cursor_list,
            &copyback_cursor_list, main_thread_id,
            num_threads] (int64 start, int64 limit) {
          int64 thread_id = Env::Default()->GetCurrentThreadId();
          int position;
          if (thread_id == main_thread_id) {
            position = num_threads;
          } else {
            auto iter = hash_map_.find_wait_free(thread_id);
            if (iter.first == -1) {
            // bind a new thread to a local cursor_list
              position = thread_id % num_threads;
              while (!__sync_bool_compare_and_swap(&(occupy_flag_[position]),
                                                   false, true)) {
                position = (position + 1) % num_threads;
            }
              hash_map_.insert_lockless(
                        std::move(std::pair<int64, int>(thread_id, position)));
            } else {
              position = iter.second;
            }
          }
          ev->LookupWithFreqBatch(indices_flat.data(), memcpy_address,
                                  start, limit, init_cursor_list[position],
                                  copyback_cursor_list[position]);
        };
        Shard(worker_threads->num_threads, worker_threads->workers, indices_size,
            slice_bytes, do_work);
        for (int i = 1; i < worker_threads->num_threads + 1; i++) {
          if (init_cursor_list[i].size()>0) {
            init_cursor_list[0].splice(init_cursor_list[0].end(),
                                       init_cursor_list[i]);
          }
          if (copyback_cursor_list[i].size()>0) {
            copyback_cursor_list[0].splice(copyback_cursor_list[0].end(),
                                           copyback_cursor_list[i]);
          }
        }
        //Pointers in memcpy_address here will 
        //be cast to ValuePtr<Tvalue>* in this funcation.
        ev->AllocateMemoryForNewFeatures(
            memcpy_address,
            init_cursor_list[0]);

        ev->SetDefaultValueOfNewFeatures(
            indices_flat.data(), indices_size,
            init_cursor_list[0], memcpy_address,
            default_v, get_default_v_fn_);

        ev->CopyEmbeddingsFromCPUToGPU(
            indices_flat.data(),
            copyback_cursor_list[0],
            memcpy_address);

        ev->CopyEmbeddingsToBuffer(
            out_base, indices_size,
            slice_elems, memcpy_address);
        delete []memcpy_address;
      } else {
        auto do_work = [this, indices_flat,
             out_base, slice_elems, c, default_v, ev, counts] (
                 int64 start, int64 limit) {
          for (int64 i = start; i < limit; ++i) {
            TValue* default_v_ptr = get_default_v_fn_(
                default_v, indices_flat(i), i, ev->GetDefaultValueDim(),
                ev->ValueLen());
            int32 count = get_count_fn_(counts, i);
            ev->LookupOrCreate(indices_flat(i),
                out_base + i * slice_elems, default_v_ptr, count);
          }
        };
        auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
        Shard(worker_threads->num_threads,
              worker_threads->workers, indices_size,
              slice_bytes, do_work);
      }

      if (ev->IsMultiLevel()) {
        ev->storage_manager()->Schedule([ev, indices]() {
          embedding::BatchCache<TKey>* cache = ev->Cache();
          cache->add_to_rank(indices);
        });
      }
    }
  }

  private:
    bool is_use_default_value_tensor_;
    std::function<
      TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
    std::function<int32(int32*, int64)> get_count_fn_;
    typedef google::dense_hash_map_lockless<int64, int> LockLessHashMap;
    LockLessHashMap hash_map_;
    bool* occupy_flag_ = nullptr;
    mutex m_init_occupy_flag_;
};

#define REGISTER_GATHER_FULL(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGather")                \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("default_value")        \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherGPUOp<ktype, vtype>)
#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, int32, type);      \
  REGISTER_GATHER_FULL(dev, int64, type)
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)
TF_CALL_float(REGISTER_GATHER_GPU);
TF_CALL_double(REGISTER_GATHER_GPU);
#undef REGISTER_GATHER_GPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

#define REGISTER_GATHER_FULL(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGatherV1")              \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("default_value")        \
                              .HostMemory("counts")               \
                              .HostMemory("output")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOp<ktype, vtype>)

#define REGISTER_GATHER_ALL_INDICES(type)                         \
  REGISTER_GATHER_FULL(CPU, int32, type);                         \
  REGISTER_GATHER_FULL(CPU, int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_GATHER_ALL_INDICES)
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL
/*
// Op that outputs tensors of all keys and all values.
template <typename TKey, typename TValue>
class KvResourceImportOp : public OpKernel {
 public:
  explicit KvResourceImportOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES(c, steps_to_live_ >= 0,
                 errors::InvalidArgument(
                    "steps_to_live must >= 0, ", std::to_string(steps_to_live_)));
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, dtype_ == context->input(1).dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    dtype_, " and ", context->input(1).dtype()));
    EmbeddingVar<TKey, TValue>* variable = nullptr;
    const Tensor& default_values = context->input(1);
    OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, HandleFromInput(context, 0), &variable,
            [this, default_values](EmbeddingVar<TKey, TValue>** ptr) {
              auto ht = KVFactory<TKey, TValue>::CreateKV(
                  ht_type_, ht_partition_num_);
              *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar",
                       new HashMap<TKey, TValue>(
                         ht, cpu_allocator()),
                         steps_to_live_);
              return (*ptr)->Init(default_values);
            }));
    core::ScopedUnref unref_me(variable);

    HashMap<TKey, TValue>* hashmap = variable->hashmap();
    const Tensor& keys = context->input(3);
    const Tensor& values = context->input(4);
    const Tensor& versions = context->input(5);
    LOG(INFO) <<  "EV:"
              << HandleFromInput(context, 0).name()
              << ", Import Size:"
              <<  keys.dim_size(0);
    OP_REQUIRES_OK(context, hashmap->Import(keys, values, versions));
    variable->SetInitialized();
  }

 private:
  DataType dtype_;
  TensorShape shape_;
  int64 steps_to_live_;
  std::string ht_type_;
  int64 ht_partition_num_;
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImport")             \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_ALL_TYPES(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS
*/

constexpr int64 DEFAULT_RESTORE_THREAD_NUM = 4;

class KvRestoreThreadPool {
 public:
  KvRestoreThreadPool() {
    TF_CHECK_OK(ReadInt64FromEnvVar("TF_EV_RESTORE_THREAD_NUM",
          DEFAULT_RESTORE_THREAD_NUM, &thread_num_));
  }

  static thread::ThreadPool* GetInstance() {
    static thread::ThreadPool tp(Env::Default(),
        "restore_ev_threadpool", thread_num_);
    return &tp;
  }

 private:
  static int64 thread_num_;
};

int64 KvRestoreThreadPool::thread_num_ =
    DEFAULT_RESTORE_THREAD_NUM;

template <typename TKey, typename TValue>
class KvResourceImportV2Op: public AsyncOpKernel {
 public:
  explicit KvResourceImportV2Op(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES(c, steps_to_live_ >= 0,
                 errors::InvalidArgument(
                    "steps_to_live must >= 0, ",
                    std::to_string(steps_to_live_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                 errors::InvalidArgument(
                    "partition_id must >= 0, ",
                    std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                 errors::InvalidArgument(
                    "partition_num must >= 1, ",
                    std::to_string(partition_num_)));
    //OP_REQUIRES_OK(c, c->GetAttr("restore_versions", &restore_versions_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));
    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));
    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
          &false_positive_probability_));
    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold",
          &l2_weight_threshold_));
    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim",
          &default_value_dim_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_no_permission",
          &default_value_no_permission_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));
    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));
    OP_REQUIRES_OK(c, c->GetAttr("storage_size", &storage_size_));
    OP_REQUIRES_OK(c, c->GetAttr("record_freq", &record_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("record_version", &record_version_));
    OP_REQUIRES_OK(c, c->GetAttr("reset_version", &reset_version_));

    if ((filter_freq_ != 0 && max_element_size_ == 0)
         || steps_to_live_ != -1 || record_freq_
         || record_version_ || storage_type > 5) {
      if (block_num_ > 1 || (filter_freq_ != 0 && storage_type <= 5)) {
        layout_ = "normal";
      } else {
        layout_ = "normal_contiguous";
      }
    } else {
      layout_ = "light";
    }

    CHECK(block_num_ == 1 || layout_ != "normal_contiguous");

    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EV_ASYNC_RESTORE", true,
                                   &ev_async_restore_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(4);
    const std::string name_string = name.scalar<string>()();
    const Tensor& default_values = context->input(3);
    OP_REQUIRES(context, dtype_ == default_values.dtype(),
        errors::InvalidArgument(
          "Variable and ddd value dtypes don't match; respectively, ",
          dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 1);
    ResourceHandle handle_primary = HandleFromInput(context, 2);
    std::string opname = handle_self.name();
    EmbeddingVar<TKey, TValue>* ev = nullptr;

    if (handle_self.name() == handle_primary.name() &&
         handle_self.container() == handle_primary.container()) {
      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname,
             handle_self](EmbeddingVar<TKey, TValue>** ptr) {
              auto storage_manager =
                new embedding::StorageManager<TKey, TValue>(
                  handle_self.name(), embedding::StorageConfig(
                    storage_type_, storage_path_, storage_size_, layout_));
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                         storage_manager,
                         EmbeddingConfig(
                           emb_index_ + block_num_ * slot_index_,
                           emb_index_,
                           block_num_, slot_num_, opname + "-primary",
                           steps_to_live_, filter_freq_,
                           max_freq_, l2_weight_threshold_,
                           layout_,  max_element_size_,
                           false_positive_probability_,
                           counter_type_, default_value_dim_,
                           default_value_no_permission_,
                           record_freq_, record_version_));
             return Status::OK();
            }));
      ev->Init(default_values, default_value_dim_);
    } else {
      EmbeddingVar<TKey, TValue>* primary_variable = nullptr;
      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname,
            handle_primary](EmbeddingVar<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             auto storage_manager =
               new embedding::StorageManager<TKey, TValue>(
                 handle_primary.name(), embedding::StorageConfig(
                   storage_type_, storage_path_, storage_size_,
                   layout_));
             *ptr = new EmbeddingVar<TKey, TValue>(handle_primary.name(),
                 storage_manager, EmbeddingConfig(
                   primary_emb_index + block_num_ * primary_slot_index,
                   primary_emb_index, block_num_, slot_num_,
                   opname + "-primary", steps_to_live_, filter_freq_,
                   max_freq_, l2_weight_threshold_,
                   layout_,  max_element_size_,
                   false_positive_probability_,
                   counter_type_, 0, record_freq_, record_version_));
            // default_values is slot value, should not to initialize primary value
            return Status::OK();
           }));

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, primary_variable,
             handle_self](EmbeddingVar<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                  primary_variable->storage_manager(),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                    emb_index_, block_num_, slot_num_, opname,
                    steps_to_live_, filter_freq_, max_freq_,
                    l2_weight_threshold_, layout_, max_element_size_,
                    false_positive_probability_,
                    counter_type_, default_value_dim_,
                    default_value_no_permission_,
                    record_freq_, record_version_));
             return (*ptr)->Init(default_values, default_value_dim_);
            }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);

    auto do_compute = [this, context, file_name_string, ev,
         name_string, done] () {
      BundleReader reader(Env::Default(), file_name_string);
      auto s = reader.status();
      if (!s.ok()) {
        LOG(FATAL) << "Restore EV failure, create BundleReader error:"
                   << s.ToString();
      }

      EVRestoreDynamically(
          ev, name_string, partition_id_, partition_num_, context, &reader,
          "-partition_offset", "-keys", "-values", "-versions", "-freqs",
          reset_version_);
      ev->SetInitialized();
      done();
    };

    if (ev_async_restore_) {
      auto tp = KvRestoreThreadPool::GetInstance();
      tp->Schedule(do_compute);
    } else {
      do_compute();
    }
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  DataType counter_type_;
  int64 max_element_size_;
  float false_positive_probability_;
  TensorShape shape_;
  int64 steps_to_live_;
  bool restore_versions_;
  string ht_type_;
  int64 ht_partition_num_;
  int64 emb_index_;
  int64 slot_index_;
  int64 block_num_;
  int64 slot_num_;
  int64 filter_freq_;
  float l2_weight_threshold_;
  std::string layout_;
  int64 max_freq_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  std::vector<int64> storage_size_;
  int64 default_value_dim_;
  float default_value_no_permission_;
  bool record_freq_;
  bool record_version_;
  bool reset_version_;
  bool ev_async_restore_;
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV2")           \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV2Op<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV2")           \
                            .Device(DEVICE_GPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV2Op<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA


template <typename TKey, typename TValue>
class KvResourceImportV3Op: public AsyncOpKernel {
 public:
  explicit KvResourceImportV3Op(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                 errors::InvalidArgument(
                    "partition_id must >= 0, ",
                    std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                 errors::InvalidArgument(
                    "partition_num must >= 1, ",
                    std::to_string(partition_num_)));
    OP_REQUIRES_OK(c, c->GetAttr("reset_version", &reset_version_));

    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EV_ASYNC_RESTORE", true,
                                   &ev_async_restore_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(2);
    const std::string name_string = name.scalar<string>()();

    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 1), &ev));

    core::ScopedUnref unref_me(ev);

    auto do_compute = [this, context, file_name_string, ev,
         name_string, done] () {
      BundleReader reader(Env::Default(), file_name_string);
      auto s = reader.status();
      if (!s.ok()) {
        LOG(FATAL) << "Restore EV failure, create BundleReader error:"
                   << s.ToString();
      }

      std::string name_string_temp(name_string);
      std::string new_str = "_";
      int64 pos = name_string_temp.find("/");
      while (pos != std::string::npos) {
        name_string_temp.replace(pos, 1, new_str.data(), 1);
        pos = name_string_temp.find("/");
      }
      std::string ssd_record_file_name =
          file_name_string + "-" + name_string_temp + "-ssd_record";
      //TODO: support change the partition number
      if (Env::Default()->FileExists(ssd_record_file_name + ".index").ok()) {
        std::string ssd_emb_file_name =
            file_name_string + "-" + name_string_temp + "-emb_files";
        if (ev->IsUsePersistentStorage()) {
          RestoreSsdRecord(ev, ssd_record_file_name, ssd_emb_file_name);
        } else {
          LoadSsdData(ev, ssd_record_file_name, ssd_emb_file_name);
        }
      }

      EVRestoreDynamically(
          ev, name_string, partition_id_, partition_num_, context, &reader,
          "-partition_offset", "-keys", "-values", "-versions", "-freqs",
          reset_version_);
      ev->SetInitialized();
      done();
    };

    if (ev_async_restore_) {
      auto tp = KvRestoreThreadPool::GetInstance();
      tp->Schedule(do_compute);
    } else {
      do_compute();
    }
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  TensorShape shape_;
  bool reset_version_;
  bool ev_async_restore_;
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV3")           \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV3Op<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV3")           \
                            .Device(DEVICE_GPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV3Op<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA


template <typename TKey, typename TValue>
class KvResourceIncrImportOp: public AsyncOpKernel {
 public:
  explicit KvResourceIncrImportOp(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));

    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
        errors::InvalidArgument(
          "partition_id must >= 0, ", std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
        errors::InvalidArgument(
          "partition_num must >= 1, ", std::to_string(partition_num_)));

  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(2);
    const std::string name_string = name.scalar<string>()();

    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(context,
        LookupResource(context, HandleFromInput(context, 1), &ev));

    core::ScopedUnref unref_me(ev);

    BundleReader reader(Env::Default(), file_name_string);
    OP_REQUIRES_OK(context, reader.status());

    LOG(INFO) << "incr import, evname:"
              << name_string
              << "partition_num:"
              << partition_num_;

    EVRestoreDynamically(
        ev, name_string, partition_id_, partition_num_, context, &reader,
        "-incr_partition_offset", "-sparse_incr_keys", "-sparse_incr_values",
        "-sparse_incr_versions", "-sparse_incr_freqs");
    ev->SetInitialized();
    done();
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  TensorShape shape_;
  int64 steps_to_live_;
  bool restore_versions_;
  string ht_type_;
  int64 ht_partition_num_;
};


#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceIncrImport")         \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceIncrImportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceIncrImport")         \
                            .Device(DEVICE_GPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceIncrImportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

// Op that outputs tensors of all keys and all values.
template<typename TKey, typename TValue>
class KvResourceExportOp : public OpKernel {
 public:
  explicit KvResourceExportOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    EmbeddingVar<TKey, TValue> *ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    std::vector<TKey> tot_key_list;
    std::vector<TValue *> tot_valueptr_list;
    std::vector<int64> tot_version_list;
    std::vector<int64> tot_freq_list;
    embedding::Iterator* it = nullptr;
    int64 total_size = ev->GetSnapshot(
        &tot_key_list, &tot_valueptr_list, &tot_version_list,
        &tot_freq_list, &it);

    // Create an output tensor
    Tensor *keys_output_tensor = NULL;
    Tensor *values_output_tensor = NULL;
    Tensor *versions_output_tensor = NULL;
    Tensor *freq_output_tensor = NULL;

    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          0, TensorShape({total_size}), &keys_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          1, TensorShape({total_size, ev->ValueLen()}),
          &values_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          2, TensorShape({tot_version_list.size()}),
          &versions_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          3, TensorShape({tot_freq_list.size()}),
          &freq_output_tensor));

    auto keys_output = keys_output_tensor->template flat<TKey>();
    auto val_matrix = values_output_tensor->matrix<TValue>();
    auto versions_output = versions_output_tensor->template flat<int64>();
    auto freq_output = freq_output_tensor->template flat<int64>();

    for(size_t i = 0; i < total_size; i++) {
      keys_output(i) = tot_key_list[i];
      TValue *value = tot_valueptr_list[i];
      for(int64 m = 0; m < ev->ValueLen(); m++) {
        val_matrix(i, m) = *(value + m);
      }
      if (tot_version_list.size() != 0) {
        versions_output(i) = tot_version_list[i];
      }
      if (tot_freq_list.size() != 0) {
        freq_output(i) = tot_freq_list[i];
      }
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceExport")             \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("Tvalues"), \
                          KvResourceExportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceExport")             \
                            .Device(DEVICE_GPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("Tvalues"), \
                          KvResourceExportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

template<typename TKey, typename TValue>
class KvResourceGeneratePartitionedTensorOp : public OpKernel {
 public:
  explicit KvResourceGeneratePartitionedTensorOp(
      OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor& keys = ctx->input(0);
    auto keys_flat = keys.flat<TKey>();
    const Tensor& values = ctx->input(1);
    auto values_flat = values.matrix<TValue>();
    const Tensor& versions = ctx->input(2);
    auto versions_flat = versions.flat<int64>();
    const Tensor& freqs = ctx->input(3);
    auto freqs_flat = freqs.flat<int64>();

    // Create an output tensor
    Tensor *keys_output_tensor = NULL;
    Tensor *values_output_tensor = NULL;
    Tensor *versions_output_tensor = NULL;
    Tensor *freq_output_tensor = NULL;
    Tensor *partial_offset_tensor = NULL;

    std::vector<std::vector<int64>> index_list_parts;

    index_list_parts.resize(kSavedPartitionNum);

    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          0, keys.shape(), &keys_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          1, values.shape(),
          &values_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          2, versions.shape(),
          &versions_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          3, freqs.shape(),
          &freq_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
         4, TensorShape({kSavedPartitionNum + 1}),
         &partial_offset_tensor));

    auto keys_output = keys_output_tensor->template flat<TKey>();
    auto val_matrix = values_output_tensor->matrix<TValue>();
    auto versions_output = versions_output_tensor->template flat<int64>();
    auto freqs_output = freq_output_tensor->template flat<int64>();
    auto offset_output = partial_offset_tensor->template flat<int32>();
    int64 key_num = keys_flat.dimension(0);
    for (int i = 0; i < key_num; i++) {
      for (int partid = 0; partid < kSavedPartitionNum; partid++) {
        if (keys_flat(i) % kSavedPartitionNum == partid) {
            index_list_parts[partid].emplace_back(i);
            break;
        }
      }
    }
    int32 total_count = 0;
    offset_output(0) = 0;
    for (int partid = 0; partid < kSavedPartitionNum; partid++) {
      for (int i = 0; i < index_list_parts[partid].size(); i++) {
        keys_output(total_count) = keys_flat(index_list_parts[partid][i]);
        for (int j = 0; j < values_flat.dimension(1); j++) {
          val_matrix(total_count, j) =
            values_flat(index_list_parts[partid][i], j);
        }
        total_count++;
      }
      offset_output(partid + 1) = total_count;
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGeneratePartitionedTensor")             \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("Tvalues"), \
                          KvResourceGeneratePartitionedTensorOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)

REGISTER_KERNELS_ALL_INDEX(float);

#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS


#if GOOGLE_CUDA
#if TENSORFLOW_USE_GPU_EV
#define REGISTER_KV_VAR_HANDLE(ktype, vtype)                           \
  REGISTER_KERNEL_BUILDER(Name("KvVarHandleOp")                        \
                          .Device(DEVICE_GPU)                          \
                          .TypeConstraint<ktype>("Tkeys")              \
                          .TypeConstraint<vtype>("dtype"),             \
                          ResourceHandleOp<EmbeddingVarGPU<ktype, vtype>>);
REGISTER_KV_VAR_HANDLE(int32, float)
REGISTER_KV_VAR_HANDLE(int64, float)
#undef REGISTER_KV_VAR_HANDLE

template <typename T, typename TKey, typename TValue>
class KvVariableShapeOpGPU : public OpKernel {
 public:
  explicit KvVariableShapeOpGPU(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVarGPU<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    TensorShape shape({ev->Size(), ev->ValueLen()});
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {shape.dims()}, &output));
    for (int i = 0; i < shape.dims(); ++i) {
      output->flat<T>()(i) = shape.dim_size(i);
    }
  }
};

#define REGISTER_KV_VARIABLE_SHAPE(type, ktype, vtype)                \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("KvVariableShape").Device(DEVICE_GPU)                      \
                             .TypeConstraint<type>("out_type")        \
                             .TypeConstraint<ktype>("Tkeys")          \
                             .HostMemory("output"),                   \
                             KvVariableShapeOpGPU<type, ktype, vtype>);
REGISTER_KV_VARIABLE_SHAPE(int32, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int32, int64, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int64, float)
#undef REGISTER_KV_VARIABLE_SHAPE

REGISTER_KERNEL_BUILDER(Name("DestroyKvResourceOp").Device(DEVICE_GPU),
                        DestroyKvResourceOp);

template <typename TKey, typename TValue>
class InitializeKvVariableOpGPU : public OpKernel {
 public:
  explicit InitializeKvVariableOpGPU(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));

     // get ev emb_index
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
     // get ev block_num
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
     // get ev slot_index
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));

    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));

    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));

    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));

    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));

    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
          &false_positive_probability_));

    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold",
          &l2_weight_threshold_));

    OP_REQUIRES_OK(c, c->GetAttr("layout", &layout_));

    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));

    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim", &default_value_dim_));

    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));

    if (filter_freq_ < 0) {
      LOG(INFO) << "filter_freq < 0 is invalid, feature filter is disabled.";
      filter_freq_ = 0;
    }

    if (steps_to_live_ == kEmbeddingVarUseDB ||
        steps_to_live_ == kInitializableEmbeddingVarUseDB) {
      LOG(INFO) << "hashmap use db";
      //use_db_ = true;
    } else {
      OP_REQUIRES(c, steps_to_live_ >= 0,
          errors::InvalidArgument(
            "steps_to_live must >= 0, ", std::to_string(steps_to_live_)));
    }
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& default_values = context->input(2);

    OP_REQUIRES(context, dtype_ == default_values.dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 0);
    ResourceHandle handle_primary = HandleFromInput(context, 1);
    std::string opname = handle_self.name();

    EmbeddingVarGPU<TKey, TValue>* ev = nullptr;

    if (handle_self.name() == handle_primary.name() &&
        handle_self.container() == handle_primary.container()) {

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, context,
             handle_self](EmbeddingVarGPU<TKey, TValue>** ptr) {
              EmbeddingConfig config(emb_index_ + block_num_ * slot_index_,
                                     emb_index_,
                                     block_num_, slot_num_,
                                     opname + "-primary",
                                     steps_to_live_, filter_freq_, max_freq_,
                                     l2_weight_threshold_, layout_,
                                     max_element_size_,
                                     false_positive_probability_,
                                     counter_type_, default_value_dim_);
              auto alloc = context->get_allocator(AllocatorAttributes());
              auto kv = new embedding::GPUHashMapKV<TKey, TValue>(config, alloc);
              *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_self.name(),
                  kv, alloc, config);
              return (*ptr)->Init(default_values, default_value_dim_);
            }));
    } else {
      EmbeddingVarGPU<TKey, TValue>* primary_variable = nullptr;
      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname, context,
            handle_primary](EmbeddingVarGPU<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             EmbeddingConfig config(
                 primary_emb_index + block_num_ * primary_slot_index,
                 primary_emb_index,
                 block_num_, slot_num_, opname + "-primary",
                 steps_to_live_, filter_freq_, max_freq_,
                 l2_weight_threshold_, layout_,
                 max_element_size_,
                 false_positive_probability_,
                 counter_type_);
             auto alloc = context->get_allocator(AllocatorAttributes());
             auto kv = new embedding::GPUHashMapKV<TKey, TValue>(config, alloc);
             *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_primary.name(),
                 kv, alloc, config);
             return (*ptr)->Init(default_values, default_value_dim_);
           }));

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, primary_variable, context,
             handle_self](EmbeddingVarGPU<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_self.name(),
                  primary_variable->kv(),
                  context->get_allocator(AllocatorAttributes()),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                    emb_index_,
                    block_num_, slot_num_, opname,
                    steps_to_live_, 0,
                    max_freq_, l2_weight_threshold_,
                    layout_, 0, -1.0, counter_type_, default_value_dim_));
             return (*ptr)->Init(default_values, default_value_dim_);
            }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);
    if (steps_to_live_ != kEmbeddingVarUseDB) {
      ev->SetInitialized();
    }
  }

 private:
  DataType dtype_;
  DataType counter_type_;
  TensorShape shape_;
  int64 steps_to_live_;
  int64 emb_index_;
  int64 block_num_;
  int64 slot_index_;
  int64 slot_num_;
  std::string ht_type_;
  int64 ht_partition_num_;
  int64 filter_freq_;
  int64 max_freq_;
  float l2_weight_threshold_;
  std::string layout_;
  int64 max_element_size_;
  float false_positive_probability_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  int64 default_value_dim_;
};

#define REGISTER_KERNELS(ktype, vtype)                               \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableOp")             \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<ktype>("Tkeys")        \
                              .TypeConstraint<vtype>("dtype"),       \
                          InitializeKvVariableOpGPU<ktype, vtype>);
#define REGISTER_GPU_KERNELS(T)        \
  REGISTER_KERNELS(int32, T);     \
  REGISTER_KERNELS(int64, T);
TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_double(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename TKey, typename TValue>
class KvResourceGatherOpGPU : public OpKernel {
 public:
  explicit KvResourceGatherOpGPU(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVarGPU<TKey, TValue>* ev = nullptr;
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
      const int64 indices_size = static_cast<int64>(
          indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      OP_REQUIRES(c, ev->ValueLen() == slice_elems,
          errors::InvalidArgument(
            "ev's value_len should same with output's dimension(1)",
            std::to_string(slice_elems), std::to_string(ev->ValueLen())));

      const TKey* key_base = &indices_flat(0);
      const Device& device = c->eigen_device<Device>();
      if (is_use_default_value_tensor_) {
        Tensor default_values(c->input(2));
        auto default_value_num = default_values.NumElements() / ev->ValueLen();
        auto default_values_matrix = default_values.shaped<TValue, 2>(
            {default_value_num, ev->ValueLen()});     
        TValue* default_v_base = &default_values_matrix(0, 0);
        ev->LookupOrCreate(key_base, out_base, default_v_base,
            default_value_num, is_use_default_value_tensor_,
            indices_size, device);
      } else {
        ev->LookupOrCreate(key_base, out_base, ev->GetDefaultValuePtr(),
            ev->GetDefaultValueDim(), is_use_default_value_tensor_,
            indices_size, device);
      }
    }
  }

  private:
    bool is_use_default_value_tensor_;
};

#define REGISTER_GATHER_FULL(dev, ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGather")                \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOpGPU<GPUDevice, ktype, vtype>)
#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, int32, type);      \
  REGISTER_GATHER_FULL(dev, int64, type)
#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(GPU, type)
TF_CALL_float(REGISTER_GATHER_GPU);
TF_CALL_double(REGISTER_GATHER_GPU);
#undef REGISTER_GATHER_GPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

template<typename Device, typename TKey, typename TValue>
class KvResourceExportOpGPU : public OpKernel {
 public:
  explicit KvResourceExportOpGPU(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    EmbeddingVarGPU<TKey, TValue> *ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    int64 total_size = ev->Size();

    // Create an output tensor
    Tensor *keys_output_tensor = NULL;
    Tensor *values_output_tensor = NULL;
    Tensor *versions_output_tensor = NULL;
    Tensor *freq_output_tensor = NULL;

    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, TensorShape({total_size}),
          &keys_output_tensor));
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(1, TensorShape({total_size, ev->ValueLen()}),
          &values_output_tensor));
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(2, TensorShape({0}),
          &versions_output_tensor));
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(3, TensorShape({0}),
          &freq_output_tensor));

    auto keys_flat = keys_output_tensor->flat<TKey>();
    TKey* key_base = &keys_flat(0);
    auto values_flat = values_output_tensor->flat<TValue>();
    TValue* value_base = &values_flat(0);

    auto device = ctx->eigen_device<Device>();
    ev->GetSnapshot(key_base, value_base, device);
  }
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceExport")             \
                            .Device(DEVICE_GPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("Tvalues") \
                            .HostMemory("keys")                \
                            .HostMemory("values")              \
                            .HostMemory("versions")            \
                            .HostMemory("freqs"),              \
                          KvResourceExportOpGPU<GPUDevice, ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)

REGISTER_KERNELS_ALL_INDEX(float);

#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceImportV2OpGPU: public OpKernel {
 public:
  explicit KvResourceImportV2OpGPU(OpKernelConstruction* c)
      : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES(c, steps_to_live_ >= 0,
                 errors::InvalidArgument(
                    "steps_to_live must >= 0, ",
                    std::to_string(steps_to_live_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                 errors::InvalidArgument(
                    "partition_id must >= 0, ",
                    std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                 errors::InvalidArgument(
                    "partition_num must >= 1, ",
                    std::to_string(partition_num_)));
    //OP_REQUIRES_OK(c, c->GetAttr("restore_versions", &restore_versions_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));
    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));
    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
          &false_positive_probability_));
    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold",
          &l2_weight_threshold_));
    OP_REQUIRES_OK(c, c->GetAttr("layout", &layout_));
    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim",
          &default_value_dim_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));
    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));
    OP_REQUIRES_OK(c, c->GetAttr("storage_size", &storage_size_));
    OP_REQUIRES_OK(c, c->GetAttr("record_freq", &record_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("record_version", &record_version_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(4);
    const std::string name_string = name.scalar<string>()();
    const Tensor& default_values = context->input(3);
    OP_REQUIRES(context, dtype_ == default_values.dtype(),
        errors::InvalidArgument(
          "Variable and ddd value dtypes don't match; respectively, ",
          dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 1);
    ResourceHandle handle_primary = HandleFromInput(context, 2);
    std::string opname = handle_self.name();
    EmbeddingVarGPU<TKey, TValue>* ev = nullptr;
    if (handle_self.name() == handle_primary.name() &&
        handle_self.container() == handle_primary.container()) {
      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, context,
             handle_self](EmbeddingVarGPU<TKey, TValue>** ptr) {
              EmbeddingConfig config(
                               emb_index_ + block_num_ * slot_index_,
                               emb_index_,
                               block_num_, slot_num_,
                               opname + "-primary",
                               steps_to_live_, filter_freq_, max_freq_,
                               l2_weight_threshold_, layout_,
                               max_element_size_,
                               false_positive_probability_,
                               counter_type_, default_value_dim_);
              auto alloc = context->get_allocator(AllocatorAttributes());
              auto kv = new embedding::GPUHashMapKV<TKey, TValue>(config, alloc);
              *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_self.name(),
                  kv, alloc, config);
              return (*ptr)->Init(default_values, default_value_dim_);
            }));
    } else {
      EmbeddingVarGPU<TKey, TValue>* primary_variable = nullptr;
      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname, context,
            handle_primary](EmbeddingVarGPU<TKey, TValue>** ptr) {
              int64 primary_slot_index(0), primary_emb_index(0);
              EmbeddingConfig config(
                  primary_emb_index + block_num_ * primary_slot_index,
                  primary_emb_index,
                  block_num_, slot_num_, opname + "-primary",
                  steps_to_live_, filter_freq_, max_freq_,
                  l2_weight_threshold_, layout_,
                  max_element_size_,
                  false_positive_probability_,
                  counter_type_);
              auto alloc = context->get_allocator(AllocatorAttributes());
              auto kv = new embedding::GPUHashMapKV<TKey, TValue>(config, alloc);
              *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_primary.name(),
                  kv, alloc, config);
              return (*ptr)->Init(default_values, default_value_dim_);
            }));

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVarGPU<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, primary_variable, context,
             handle_self](EmbeddingVarGPU<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVarGPU<TKey, TValue>(handle_self.name(),
                  primary_variable->kv(),
                  context->get_allocator(AllocatorAttributes()),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                    emb_index_,
                    block_num_, slot_num_, opname,
                    steps_to_live_, 0,
                    max_freq_, l2_weight_threshold_,
                    layout_, 0, -1.0, counter_type_, default_value_dim_));
             return (*ptr)->Init(default_values, default_value_dim_);
            }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);

    BundleReader reader(Env::Default(), file_name_string);
    auto s = reader.status();
    if (!s.ok()) {
      LOG(FATAL) << "Restore EV failure, create BundleReader error:"
                 << s.ToString();
    }

    EVRestoreDynamicallyGPU(
        ev, name_string, partition_id_, partition_num_, context, &reader,
        "-partition_offset", "-keys", "-values", "-versions", "-freqs");
    ev->SetInitialized();
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  DataType counter_type_;
  int64 max_element_size_;
  float false_positive_probability_;
  TensorShape shape_;
  int64 steps_to_live_;
  bool restore_versions_;
  string ht_type_;
  int64 ht_partition_num_;
  int64 emb_index_;
  int64 slot_index_;
  int64 block_num_;
  int64 slot_num_;
  int64 filter_freq_;
  float l2_weight_threshold_;
  std::string layout_;
  int64 max_freq_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  std::vector<int64> storage_size_;
  int64 default_value_dim_;
  bool record_freq_;
  bool record_version_;
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV2")           \
                            .Device(DEVICE_GPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV2OpGPU<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX);
//TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#endif  // TENSORFLOW_USE_GPU_EV
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

#define REGISTER_EV_GET_FREQUENCY(ktype, vtype)                 \
  REGISTER_KERNEL_BUILDER(Name("EVGetFrequency")                \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("Tvalues"),  \
                          EVGetFrequencyOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                        \
  REGISTER_EV_GET_FREQUENCY(int32, type)                        \
  REGISTER_EV_GET_FREQUENCY(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_EV_GET_FREQUENCY

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

#define REGISTER_EV_GET_VERSION(ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("EVGetVersion")                  \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("Tvalues"),  \
                          EVGetVersionOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                        \
  REGISTER_EV_GET_VERSION(int32, type)                          \
  REGISTER_EV_GET_VERSION(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_EV_GET_VERSION

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
      int v = ev->storage_manager()->LookupTier(indices_flat(i));
      output->flat<int>()(i) = v;
    }
  }
};

#define REGISTER_EV_LOOKUP_TIER(ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvResourceLookupTier")          \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          KvResourceLookupTierOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                        \
  REGISTER_EV_LOOKUP_TIER(int32, type)                          \
  REGISTER_EV_LOOKUP_TIER(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_EV_LOOKUP_TIER

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#define REGISTER_EV_LOOKUP_TIER(ktype, vtype)                   \
  REGISTER_KERNEL_BUILDER(Name("KvResourceLookupTier")          \
                            .Device(DEVICE_GPU)                 \
                            .HostMemory("ids")                  \
                            .HostMemory("output")               \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          KvResourceLookupTierOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                        \
  REGISTER_EV_LOOKUP_TIER(int32, type)                          \
  REGISTER_EV_LOOKUP_TIER(int64, type)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_EV_LOOKUP_TIER
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

}  // namespace tensorflow


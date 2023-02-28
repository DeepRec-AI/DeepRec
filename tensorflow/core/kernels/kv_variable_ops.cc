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
}

#define REGISTER_KV_VAR_HANDLE(dev, ktype, vtype)                      \
  REGISTER_KERNEL_BUILDER(Name("KvVarHandleOp")                        \
                          .Device(DEVICE_##dev)                        \
                          .TypeConstraint<ktype>("Tkeys")              \
                          .TypeConstraint<vtype>("dtype"),             \
                          ResourceHandleOp<EmbeddingVar<ktype, vtype>>);
#define REGISTER_KERNELS_ALL(dev, type)                                \
  REGISTER_KV_VAR_HANDLE(dev, int32, type)                             \
  REGISTER_KV_VAR_HANDLE(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KV_VAR_HANDLE

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

#define REGISTER_KERNELS(dev, type, ktype, vtype)                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("KvVariableShape").Device(DEVICE_##dev)                    \
                             .TypeConstraint<type>("out_type")        \
                             .TypeConstraint<ktype>("Tkeys")          \
                             .TypeConstraint<vtype>("dtype")          \
                             .HostMemory("output"),                   \
                             KvVariableShapeOp<type, ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)                               \
  REGISTER_KERNELS(dev, int32, int32, type)                           \
  REGISTER_KERNELS(dev, int32, int64, type)                           \
  REGISTER_KERNELS(dev, int64, int32, type)                           \
  REGISTER_KERNELS(dev, int64, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
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
    auto device_type_str = c->device_type().type_string();
    if (storage_type_ == embedding::DEFAULT) {
      if (device_type_str == "CPU") {
        storage_type_ = embedding::DRAM;
      } else {
        storage_type_ = embedding::HBM;
      }
    }

    bool if_op_on_gpu = (device_type_str == "GPU");
    bool if_embedding_on_hbm = (storage_type_ == embedding::HBM ||
                                storage_type_ == embedding::HBM_DRAM ||
                                storage_type_ == embedding::HBM_DRAM_SSDHASH);
    OP_REQUIRES(c, if_op_on_gpu == if_embedding_on_hbm,
        errors::InvalidArgument("Storage of EV and device of Op mismatch."));

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));
    OP_REQUIRES_OK(c, c->GetAttr("storage_size", &storage_size_));

    if (filter_freq_ < 0) {
      LOG(INFO) << "filter_freq < 0 is invalid, feature filter is disabled.";
      filter_freq_ = 0;
    }

    OP_REQUIRES_OK(c, c->GetAttr("layout", &layout_));
    if (!layout_.empty()) {
      // use layout by user configuration
    } else if ((filter_freq_ != 0 && max_element_size_ == 0)
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

    if ("compact" == layout_) {
      OP_REQUIRES(c, shape_.dim_size(0) == 1 &&
            storage_type_ == embedding::StorageType::DRAM,
          errors::InvalidArgument("embedding_dim must be 1 and storage type"
                                  " should be DRAM when layout is 'compact'."));
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
                  //context->get_allocator(AllocatorAttributes());
              auto embedding_config = EmbeddingConfig(
                  emb_index_ + block_num_ * slot_index_,
                  emb_index_, block_num_, slot_num_,
                  opname + "-primary", steps_to_live_,
                  filter_freq_, max_freq_,
                  l2_weight_threshold_, layout_,
                  max_element_size_, false_positive_probability_,
                  counter_type_, default_value_dim_,
                  default_value_no_permission_,
                  record_freq_, record_version_);
              auto storage_manager =
                  new embedding::StorageManager<TKey, TValue>(
                    handle_self.name(),
                    embedding::StorageConfig(
                      storage_type_, storage_path_, storage_size_, layout_,
                      embedding_config),
                    gpu_allocator);
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                         storage_manager,
                         embedding_config,
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
             //Allocator* gpu_allocator = context->get_allocator(AllocatorAttributes());
             auto embedding_config = EmbeddingConfig(
                 primary_emb_index + block_num_ * primary_slot_index,
                 primary_emb_index,
                 block_num_, slot_num_, opname + "-primary",
                 steps_to_live_, filter_freq_, max_freq_,
                 l2_weight_threshold_, layout_,
                 max_element_size_, false_positive_probability_,
                 counter_type_, 0, record_freq_, record_version_);
             auto storage_manager =
               new embedding::StorageManager<TKey, TValue>(
                 handle_primary.name(), embedding::StorageConfig(storage_type_,
                     storage_path_, storage_size_, layout_, embedding_config),
                     gpu_allocator);
             *ptr = new EmbeddingVar<TKey, TValue>(handle_primary.name(),
                        storage_manager,
                        embedding_config,
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
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(ktype, vtype)                               \
  REGISTER_KERNEL_BUILDER(Name("InitializeKvVariableOp")             \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<ktype>("Tkeys")        \
                              .TypeConstraint<vtype>("dtype"),       \
                          InitializeKvVariableOp<ktype, vtype>);

#define REGISTER_GPU_KERNELS(type)        \
  REGISTER_KERNELS(int32, type);          \
  REGISTER_KERNELS(int64, type);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS
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
#define REGISTER_KERNELS(dev, ktype, vtype)                        \
  REGISTER_KERNEL_BUILDER(Name("KvVarIsInitializedOp")             \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .TypeConstraint<vtype>("dtype")          \
                          .HostMemory("is_initialized")            \
                          .Device(DEVICE_##dev),                   \
                          KvResourceIsInitializedOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)                            \
  REGISTER_KERNELS(dev, int32, type)                               \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceInitCacheStrategyOp : public OpKernel {
 public:
  explicit KvResourceInitCacheStrategyOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("cache_strategy", &cache_strategy_));
  }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    ev->InitCache(static_cast<embedding::CacheStrategy>(cache_strategy_));
  }

 private:
  int cache_strategy_;
};

#define REGISTER_KERNELS(dev, ktype, vtype)                        \
  REGISTER_KERNEL_BUILDER(Name("KvResourceInitCacheStrategyOp")    \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .TypeConstraint<vtype>("dtype")          \
                          .Device(DEVICE_##dev),                   \
                          KvResourceInitCacheStrategyOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)                            \
  REGISTER_KERNELS(dev, int32, type)                               \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
}  // namespace tensorflow


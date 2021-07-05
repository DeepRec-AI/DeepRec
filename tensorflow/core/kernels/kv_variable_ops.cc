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
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {  
const int64 kEmbeddingVarUseDB = -214;
const int64 kInitializableEmbeddingVarUseDB = -215;
}

#define REGISTER_KV_VAR_HANDLE(ktype, vtype)                           \
  REGISTER_KERNEL_BUILDER(Name("KvVarHandleOp")                        \
                          .Device(DEVICE_CPU)                          \
                          .TypeConstraint<ktype>("Tkeys")              \
                          .TypeConstraint<vtype>("dtype"),             \
                          ResourceHandleOp<EmbeddingVar<ktype, vtype>>);
REGISTER_KV_VAR_HANDLE(int32, float)
REGISTER_KV_VAR_HANDLE(int64, float)
#undef REGISTER_KV_VAR_HANDLE

template <typename TIndex>
class ReadKvVariableOp : public OpKernel {
 public:
  explicit ReadKvVariableOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TIndex, float>* variable = nullptr;
    ResourceHandle handle = HandleFromInput(ctx, 0);
    const auto status = LookupResource(ctx, handle, &variable);
    OP_REQUIRES(ctx, status.ok(),
                errors::NotFound(
                    "Error while reading resource variable ", handle.name(),
                    " from Container: ", handle.container(),
                    ". This could mean that the variable was not initialized. ",
                    status.ToString()));

    core::ScopedUnref unref_me(variable);
    HashMap<TIndex, float>* hashmap = variable->hashmap();
    TensorShape value_shape({hashmap->ValueLen()});
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("value", value_shape, &out));
    //hashmap->Lookup(0, out);
    OP_REQUIRES(
        ctx, dtype_ == out->dtype(),
        errors::InvalidArgument(
            "Trying to read variable with wrong dtype. Expected ",
            DataTypeString(dtype_), " got ", DataTypeString(out->dtype())));
  }

 private:
  DataType dtype_;
};

#define REGISTER_READ_KV_VARIABLE(dev, type)                      \
  REGISTER_KERNEL_BUILDER(Name("ReadKvVariableOp")                \
                          .Device(DEVICE_##dev)                   \
                          .TypeConstraint<type>("Tkeys"),         \
                          ReadKvVariableOp<type>);
REGISTER_READ_KV_VARIABLE(CPU, int64)
REGISTER_READ_KV_VARIABLE(CPU, int32)
#undef REGISTER_READ_KV_VARIABLE

template <typename T, typename TKey, typename TValue>
class KvVariableShapeOp : public OpKernel {
 public:
  explicit KvVariableShapeOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* variable = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &variable));
    core::ScopedUnref unref_me(variable);
    HashMap<TKey, TValue>* hashmap = variable->hashmap();
    TensorShape shape({hashmap->Size(), hashmap->ValueLen()});
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {shape.dims()}, &output));
    for (int i = 0; i < shape.dims(); ++i) {
      output->flat<T>()(i) = shape.dim_size(i);
    }
  }
};

#define REGISTER_KV_VARIABLE_SHAPE(type, ktype, vtype)                \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("KvVariableShape").Device(DEVICE_CPU)                      \
                             .TypeConstraint<type>("out_type")        \
                             .TypeConstraint<ktype>("Tkeys"),         \
                             KvVariableShapeOp<type, ktype, vtype>);
REGISTER_KV_VARIABLE_SHAPE(int32, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int32, int64, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int64, float)
#undef REGISTER_KV_VARIABLE_SHAPE

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
  explicit InitializeKvVariableOp(OpKernelConstruction* c) : OpKernel(c), use_db_(false) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
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
    if (steps_to_live_ == kEmbeddingVarUseDB ||
        steps_to_live_ == kInitializableEmbeddingVarUseDB) {
      LOG(INFO) << "hashmap use db";
      use_db_ = true;
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

    EmbeddingVar<TKey, TValue>* variable = nullptr;

    const Tensor& slotnum = context->input(4);
    int64 slotnum_op = slotnum.scalar<int64>()(); 
    
    if (handle_self.name() == handle_primary.name() && 
        handle_self.container() == handle_primary.container()) {

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &variable,
            [this, default_values, opname, slotnum_op](EmbeddingVar<TKey, TValue>** ptr) {
              auto ht = HashMapFactory<TKey, TValue>::CreateHashMap(
                  ht_type_, ht_partition_num_);
              *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar",
                       new HashMap<TKey, TValue>(
                         ht, cpu_allocator(), use_db_,
                         EmbeddingConfig(emb_index_ +  block_num_ * slot_index_, emb_index_,
                                                        block_num_, slotnum_op, opname + "-primary", 
                                                        steps_to_live_)),
                         steps_to_live_);
             return (*ptr)->Init(default_values);
            }));

    } else {
      EmbeddingVar<TKey, TValue>* primary_variable = nullptr;

      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname, slotnum_op](EmbeddingVar<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             auto ht = HashMapFactory<TKey, TValue>::CreateHashMap(
                 ht_type_, ht_partition_num_);
             *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar",
                      new HashMap<TKey, TValue>(
                        ht, cpu_allocator(), use_db_,
                        EmbeddingConfig(primary_emb_index +  block_num_ * primary_slot_index, primary_emb_index,
                                                       block_num_, slotnum_op, opname + "-primary", 
                                                       steps_to_live_)),
                        steps_to_live_);
            return (*ptr)->Init(default_values);
           }));


      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &variable,
            [this, default_values, opname, primary_variable, slotnum_op](EmbeddingVar<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar",
                       new HashMap<TKey, TValue>(
                         primary_variable->hashmap()->kv(), cpu_allocator(), use_db_,
                         EmbeddingConfig(emb_index_ +  block_num_ * slot_index_, emb_index_,
                                                        block_num_, slotnum_op, opname,
                                                        steps_to_live_)),
                         steps_to_live_);

             return (*ptr)->Init(default_values);
            }));

      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(variable);
    if (steps_to_live_ != kEmbeddingVarUseDB) {
      variable->SetInitialized();
    }
  }

 private:
  DataType dtype_;
  TensorShape shape_;
  int64 steps_to_live_;
  int64 emb_index_;
  int64 block_num_;
  int64 slot_index_;
  std::string ht_type_;
  int64 ht_partition_num_;
  bool use_db_;
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
TF_CALL_ALL_TYPES(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceIsInitializedOp : public OpKernel {
 public:
  explicit KvResourceIsInitializedOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
    EmbeddingVar<TKey, TValue>* variable = nullptr;
    bool found;
    if (LookupResource<EmbeddingVar<TKey, TValue>>(ctx, HandleFromInput(ctx, 0), &variable).ok()) {
      found = variable->IsInitialized();
      variable->Unref();
    } else {
      found = false;
    }

    output->flat<bool>()(0) = found;
  }
};
#define REGISTER_KERNELS(ktype, vtype)                             \
  REGISTER_KERNEL_BUILDER(Name("KvVarIsInitializedOp")             \
                          .TypeConstraint<ktype>("Tkeys")          \
                          .Device(DEVICE_CPU),                     \
                          KvResourceIsInitializedOp<ktype, vtype>);
REGISTER_KERNELS(int32, float)
REGISTER_KERNELS(int64, float)
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceGatherOp : public OpKernel {
 public:
  explicit KvResourceGatherOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* variable = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &variable));
    core::ScopedUnref unref_me(variable);
    HashMap<TKey, TValue>* hashmap = variable->hashmap();

    const Tensor& indices = c->input(1);
    const int64 N = indices.NumElements();

    Tensor default_values(c->input(2));
    auto default_values_matrix = default_values.shaped<TValue, 2>(
        {default_values.NumElements()/hashmap->ValueLen(), hashmap->ValueLen()});

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({hashmap->ValueLen()});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      OP_REQUIRES(c, hashmap->ValueLen() == slice_elems,
          errors::InvalidArgument(
              "hashmap's value_len should same with output's dimension(1)",
              std::to_string(slice_elems), std::to_string(hashmap->ValueLen())));
      const size_t slice_bytes = slice_elems * sizeof(TValue);
      auto do_work = [this, &indices_flat, &default_values_matrix,
           &out_base, &slice_elems, &hashmap] (int64 start, int64 limit) {
        for (int64 i = start; i < limit; ++i) {
          TValue* default_v = &default_values_matrix(i, 0);
          hashmap->LookupOrCreateHybrid(indices_flat(i),
              out_base + i * slice_elems, default_v);
        }
      };
      auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads->num_threads, worker_threads->workers, indices_size,
          slice_bytes, do_work);
    }
  }
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

#define REGISTER_GATHER_ALL_INDICES(dev, type) \
  REGISTER_GATHER_FULL(dev, int32, type);      \
  REGISTER_GATHER_FULL(dev, int64, type)

#define REGISTER_GATHER_CPU(type) REGISTER_GATHER_ALL_INDICES(CPU, type)

// Registration of the CPU implementations.
TF_CALL_ALL_TYPES(REGISTER_GATHER_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_CPU);

#undef REGISTER_GATHER_CPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL

template <typename Device, typename TKey, typename TValue, scatter_op::UpdateOp op>
class KvResourceScatterUpdateOp : public OpKernel {
 public:
  explicit KvResourceScatterUpdateOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, float>* variable = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &variable));
    core::ScopedUnref unref_me(variable);
    HashMap<TKey, float>* hashmap = variable->hashmap();
    TensorShape value_shape({hashmap->ValueLen()});
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    auto indices_flat = indices.flat<TKey>();
    auto updates_flat = updates.flat<TValue>();
    const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
    for (int64 i = 0; i < indices_size; i++) {
      Tensor param(DT_FLOAT, value_shape);
      //OP_REQUIRES_OK(c, hashmap->Lookup(indices_flat(i), &param));

      auto param_flat = param.flat<TValue>();
      // TODO(dingchen): param_size as object member
      const int64 param_size = static_cast<int64>(param_flat.dimension(0));
      int64 j = i * param_size;
      for (int64 k = 0; k < param_size; ++k) {
        param_flat(k) += updates_flat(j++);
      }
    }
  }
};

#define REGISTER_SCATTER_KERNEL_INDEX(ktype, vtype, dev, name, op)     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(name)                                                       \
          .Device(DEVICE_##dev)                                        \
          .HostMemory("resource")                                      \
          .TypeConstraint<vtype>("dtype")                              \
          .TypeConstraint<ktype>("Tkeys"),                             \
      KvResourceScatterUpdateOp<dev##Device, ktype, vtype, op>)

#define REGISTER_SCATTER_KERNEL(type, dev, name, op)           \
  REGISTER_SCATTER_KERNEL_INDEX(int32, type, dev, name, op);   \
  REGISTER_SCATTER_KERNEL_INDEX(int64, type, dev, name, op);

// TODO(apassos) add the other types here.
#define REGISTER_SCATTER_ARITHEMTIC(type, dev)                 \
  REGISTER_SCATTER_KERNEL(type, dev, "KvResourceScatterAdd",   \
                          scatter_op::UpdateOp::ADD);

// Registers CPU kernels.
#define REGISTER_SCATTER_ARITHEMTIC_CPU(type)                  \
  REGISTER_SCATTER_ARITHEMTIC(type, CPU);

TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ARITHEMTIC_CPU);

#undef REGISTER_SCATTER_ARITHEMTIC
#undef REGISTER_SCATTER_ARITHEMTIC_CPU
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX

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
              auto ht = HashMapFactory<TKey, TValue>::CreateHashMap(
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
    LOG(INFO) <<  "EV:" << HandleFromInput(context, 0).name() << ", Import Size:" <<  keys.dim_size(0);
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

template <typename TKey, typename TValue>
class KvResourceImportV2Op: public OpKernel {
 public:
  explicit KvResourceImportV2Op(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES(c, steps_to_live_ >= 0,
                 errors::InvalidArgument(
                    "steps_to_live must >= 0, ", std::to_string(steps_to_live_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                 errors::InvalidArgument(
                    "partition_id must >= 0, ", std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                 errors::InvalidArgument(
                    "partition_num must >= 1, ", std::to_string(partition_num_)));
    //OP_REQUIRES_OK(c, c->GetAttr("restore_versions", &restore_versions_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
    // get ev emb_index
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
      // get ev slot_index
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
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
    EmbeddingVar<TKey, TValue>* variable = nullptr;

    const Tensor& slotnum = context->input(6);
    int64 slotnum_op = slotnum.scalar<int64>()(); 
    
    if (handle_self.name() == handle_primary.name() && 
         handle_self.container() == handle_primary.container()) {
      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &variable,
            [this, default_values, opname, slotnum_op](EmbeddingVar<TKey, TValue>** ptr) {
              auto ht = HashMapFactory<TKey, TValue>::CreateHashMap(
                  ht_type_, ht_partition_num_);
              *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar",
                       new HashMap<TKey, TValue>(
                         ht, cpu_allocator(), false,
                         EmbeddingConfig(emb_index_ +  block_num_ * slot_index_, emb_index_,
                                                        block_num_, slotnum_op, opname + "-primary", 
                                                        steps_to_live_)),
                         steps_to_live_);
             return (*ptr)->Init(default_values);
            }));
    } else {
      EmbeddingVar<TKey, TValue>* primary_variable = nullptr;
      
      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname, slotnum_op](EmbeddingVar<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             auto ht = HashMapFactory<TKey, TValue>::CreateHashMap(
                 ht_type_, ht_partition_num_);
             *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar",
                      new HashMap<TKey, TValue>(
                        ht, cpu_allocator(), false,
                        EmbeddingConfig(primary_emb_index +  block_num_ * primary_slot_index, primary_emb_index,
                                                       block_num_, slotnum_op, opname + "-primary", 
                                                       steps_to_live_)),
                        steps_to_live_);
            return (*ptr)->Init(default_values);
           }));

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &variable,
            [this, default_values, opname, primary_variable, slotnum_op](EmbeddingVar<TKey, TValue>** ptr) {
              *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar",
                       new HashMap<TKey, TValue>(
                         primary_variable->hashmap()->kv(), cpu_allocator(), false,
                         EmbeddingConfig(emb_index_ +  block_num_ * slot_index_, emb_index_,
                                                        block_num_, slotnum_op, opname,
                                                        steps_to_live_)),
                         steps_to_live_);

             return (*ptr)->Init(default_values);
            }));

      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(variable);

    HashMap<TKey, TValue>* hashmap = variable->hashmap();
    BundleReader reader(Env::Default(), file_name_string);
    OP_REQUIRES_OK(context, reader.status());


    EVRestoreDynamically(hashmap, name_string, partition_id_, partition_num_, context, &reader, "-partition_offset", "-keys", "-values", "-versions");
    variable->SetInitialized();
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
  int64 emb_index_;
  int64 slot_index_;
  int64 block_num_;
};


#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV2")             \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV2Op<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_ALL_TYPES(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS



template <typename TKey, typename TValue>
class KvResourceIncrImportOp: public OpKernel {
 public:
  explicit KvResourceIncrImportOp(OpKernelConstruction* c) : OpKernel(c) {
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

  void Compute(OpKernelContext* context) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(2);
    const std::string name_string = name.scalar<string>()();
  
    EmbeddingVar<TKey, TValue>* variable = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 1), &variable));

   
    core::ScopedUnref unref_me(variable);

    HashMap<TKey, TValue>* hashmap = variable->hashmap();
    BundleReader reader(Env::Default(), file_name_string);
    OP_REQUIRES_OK(context, reader.status());

    LOG(INFO) << "incr import, evname:" << name_string << "partition_num:" <<partition_num_;
    EVRestoreDynamically(hashmap, name_string, partition_id_, partition_num_, context, &reader, "-incr_partition_offset", "-sparse_incr_keys", "-sparse_incr_values", "-sparse_incr_versions");
    variable->SetInitialized();
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
  REGISTER_KERNEL_BUILDER(Name("KvResourceIncrImport")             \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceIncrImportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
  REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_ALL_TYPES(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

// Op that outputs tensors of all keys and all values.
template <typename TKey>
class KvResourceExportOp : public OpKernel {
 public:
  explicit KvResourceExportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, float>* variable = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &variable));
    core::ScopedUnref unref_me(variable);
    OP_REQUIRES_OK(ctx, variable->ExportValues(ctx));
  }
};

#define REGISTER_KERNEL(ktype)                                 \
  REGISTER_KERNEL_BUILDER(Name("KvResourceExport")             \
                            .Device(DEVICE_CPU)                \
                            .TypeConstraint<ktype>("Tkeys"),   \
                          KvResourceExportOp<ktype>);
REGISTER_KERNEL(int32)
REGISTER_KERNEL(int64)
#undef REGISTER_KERNEL

// Op that outputs tensors of all keys and all values.
template <typename T, typename TIndex>
class KvResourceInsertOp : public OpKernel {
 public:
  explicit KvResourceInsertOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
  }
  void Compute(OpKernelContext* ctx) override {
    HashMap<TIndex, float>* hashmap = NULL;
    OP_REQUIRES_OK(ctx, GetInputHashMap(ctx, 0, &hashmap));
    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    const Tensor& versions = ctx->input(3);
    LOG(INFO) <<  "EV:" << HandleFromInput(ctx, 0).name() << ", Incr Import Size:" <<  keys.dim_size(0);
    OP_REQUIRES_OK(ctx, hashmap->Import(keys, values, versions));
  }
 private:
  DataType dtype_;
};
#define REGISTER_KERNELS(ktype, type)                           \
    REGISTER_KERNEL_BUILDER(Name("KvResourceInsert")             \
                                    .Device(DEVICE_CPU)                \
                                    .TypeConstraint<ktype>("Tkeys")    \
                                    .TypeConstraint<type>("dtype"),    \
                                  KvResourceInsertOp<type, ktype>);
#define REGISTER_KERNELS_ALL_INDEX(type)                       \
    REGISTER_KERNELS(int32, type)                                \
  REGISTER_KERNELS(int64, type)
TF_CALL_ALL_TYPES(REGISTER_KERNELS_ALL_INDEX);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS_ALL_INDEX);
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

}  // namespace tensorflow


/* Coperight 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/hash_table/tensible_variable.h"
#include "tensorflow/core/framework/hash_table/hash_table.h"
#include "tensorflow/core/framework/hash_table/status_collector.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/common_runtime/renamed_device.h"

namespace tensorflow {

namespace {

class TensorProducerCallFrame : public CallFrameInterface {
 public:
  TensorProducerCallFrame() {}

  size_t num_retvals() const override { return 1; }

  // Callee methods.
  Status SetRetval(int index, const Tensor& val) override {
    if (index != 0) {
      return errors::InvalidArgument("Return value ", index,
                                     " is out of range.");
    }
    retval_ = val;
    return Status::OK();
  }

  size_t num_args() const override {
    return 0;
  }

  // Callee methods.
  Status GetArg(int index, Tensor* val) const override {
    (void)val;
    return errors::InvalidArgument("Argument ", index, " is out of range.");
  }

  Tensor GetRet() {
    return retval_;
  }

 private:
  Tensor retval_;
};

struct TensorProducerContext {
  std::unique_ptr<TensorProducerCallFrame> frame;
  std::unique_ptr<FunctionLibraryRuntime::Options> f_opts;
  std::unique_ptr<ScopedStepContainer> step_container;
  std::unique_ptr<CancellationManager> cancellation_manager;
};

class TensorProducerImpl : public core::RefCounted {
 public:
  Status Init(OpKernelContext* ctx, const NameAttrList& func);
  void Produce(const TensorGenerator::Consumer& consumer);
 private:
  NameAttrList func_;
  Env* env_;
  std::function<void(std::function<void()>)> runner_;
  FunctionLibraryRuntime* lib_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionLibraryRuntime::Handle f_handle_;
};

class TensorProducer {
 public:
  TensorProducer() : impl_(nullptr) { }
  ~TensorProducer() {
    if (impl_ != nullptr) {
      impl_->Unref();
    }
  }
  Status Init(OpKernelContext* ctx, const NameAttrList& func) {
    TensorProducerImpl* impl = new TensorProducerImpl;
    Status st = impl->Init(ctx, func);
    if (!st.ok()) {
      impl->Unref();
      return st;
    }
    impl_ = impl;
    return Status::OK();
  }
  void Produce(const TensorGenerator::Consumer& consumer) {
    return impl_->Produce(consumer);
  }

 private:
  TensorProducer(const TensorProducer&) = delete;
  TensorProducerImpl* impl_;
};

Status TensorProducerImpl::Init(
    OpKernelContext* ctx, const NameAttrList& func) {
  func_ = func;
  env_ = ctx->env();
  runner_ = *ctx->runner();
  TF_RETURN_IF_ERROR(ctx->function_library()->Clone(&flib_def_, &pflr_, &lib_));
  FunctionLibraryRuntime::InstantiateOptions inst_opts;
  inst_opts.state_handle = std::to_string(random::New64());
  TF_RETURN_IF_ERROR(
      lib_->Instantiate(func_.name(), AttrSlice(&func_.attr()),
                        inst_opts, &f_handle_));
  const FunctionBody* fbody = lib_->GetFunctionBody(f_handle_);
  if (fbody == nullptr) {
    return errors::Internal("Failed to instantiate function body.");
  }
  if (fbody->ret_types.size() != 1) {
    return errors::InvalidArgument(
        "Tensor Generator Func should return 1 tensor");
  }
  return Status::OK();
}

void TensorProducerImpl::Produce(const TensorGenerator::Consumer& consumer) {
  TensorProducerContext* ctx = new TensorProducerContext;
  ctx->frame.reset(new TensorProducerCallFrame);
  ctx->f_opts.reset(new FunctionLibraryRuntime::Options);
  ctx->step_container.reset(new ScopedStepContainer(
      ctx->f_opts->step_id, [this](const string& name) {
    lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
  }));
  ctx->f_opts->step_container = ctx->step_container.get();
  ctx->f_opts->runner = &runner_;
  ctx->cancellation_manager.reset(new CancellationManager);
  ctx->f_opts->cancellation_manager = ctx->cancellation_manager.get();
  CancellationManager c_mgr;
  ctx->f_opts->cancellation_manager = &c_mgr;

  Ref();
  lib_->Run(*ctx->f_opts, f_handle_, ctx->frame.get(),
  [this, ctx, consumer] (const Status& st) {
    consumer(st, ctx->frame->GetRet());
    delete ctx;
    Unref();
  });
}

}  // namespace

class TensibleVariableInitializeOp : public AsyncOpKernel {
 public:
  explicit TensibleVariableInitializeOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("factory", &func_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("initialized", &initialized_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    TensibleVariableResource* resource;
    HashTableResource* hash_table;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &hash_table), done);
    core::ScopedUnref s1(hash_table);
    OP_REQUIRES_OK_ASYNC(
        ctx,
        LookupOrCreateResource<TensibleVariableResource>(
            ctx, HandleFromInput(ctx, 0), &resource,
            [](TensibleVariableResource** ptr) {
              *ptr = new TensibleVariableResource;
              return Status::OK();
            }), done);
    core::ScopedUnref s2(resource);
    HashTable* table = hash_table->Internal();
    OP_REQUIRES_ASYNC(
        ctx, table != nullptr,
        errors::FailedPrecondition("HashTable is not initialized"), done);

    if (resource->Internal() != nullptr) {
      std::shared_ptr<TensorProducer> producer(new TensorProducer);
      OP_REQUIRES_OK_ASYNC(ctx, producer->Init(ctx, func_), done);
      resource->Internal()->GetGenerator()->SetProducer(
          [producer] (const TensorGenerator::Consumer& consumer) {
            producer->Produce(consumer);});
      resource->SetInitialized(initialized_);
      done();
      return;
    } else {
      std::shared_ptr<TensorProducer> producer(new TensorProducer);
      OP_REQUIRES_OK_ASYNC(ctx, producer->Init(ctx, func_), done);
      TensorGenerator* generator = new TensorGenerator(
          [producer] (const TensorGenerator::Consumer& consumer) {
            producer->Produce(consumer);});
      OP_REQUIRES_OK_ASYNC(
          ctx, 
          resource->CreateInternal(generator, shape_, dtype_), 
          done);
      hash_table->Ref();
      resource->Ref();
      table->AddTensible(
          resource->Internal(),
          [this, hash_table, resource, ctx, done] (Status st) {
            resource->SetInitialized(initialized_);
            hash_table->Unref();
            resource->Unref();
            OP_REQUIRES_OK_ASYNC(ctx, st, done);
            done();});
    }
  }

 private:
  NameAttrList func_;
  TensorShape shape_;
  DataType dtype_;
  bool initialized_;
};

class TensibleVariableIsInitializedOp : public OpKernel {
 public:
  explicit TensibleVariableIsInitializedOp(OpKernelConstruction* c)
    : OpKernel(c) {}

  void Compute(OpKernelContext* context) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<bool, 0>();
    TensibleVariableResource* resource = nullptr;
    Status s = LookupResource(context, HandleFromInput(context, 0), &resource);
    if (!s.ok()) {
      output_tensor() = false;
      return;
    }
    core::ScopedUnref su(resource);
    output_tensor() = resource->Initialized();
  }
};

static constexpr int kIdBlockSize = 4096;

static void BuildGatherShape(
    const TensorShape& var, const TensorShape& ids,
    TensorShape* shape, int64* slice, int64* size) {
  std::vector<int64> out_dims;
  int64 osize = 1, oslice = 1;
  for (int i = 0; i < ids.dims(); i++) {
    out_dims.push_back(ids.dim_size(i));
    osize *= ids.dim_size(i);
  }
  for (int i = 1; i < var.dims(); i++) {
    out_dims.push_back(var.dim_size(i));
    oslice *= var.dim_size(i);
  }
  *shape = TensorShape(out_dims);
  *slice = oslice;
  *size = osize;
}

template<typename T>
class TensibleVariableGatherOp : public AsyncOpKernel {
 public:
  explicit TensibleVariableGatherOp(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    TensibleVariableResource* resource;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource), done);
    core::ScopedUnref s(resource);
    TensibleVariable* tensible_variable = resource->Internal();
    OP_REQUIRES_ASYNC(
        ctx, tensible_variable != nullptr,
        errors::FailedPrecondition("TensibleVariable is not initialized"), done);
    OP_REQUIRES_ASYNC(
        ctx, DataTypeToEnum<T>::value == tensible_variable->dtype(),
        errors::FailedPrecondition("TensibleVariable dtype mismatch"), done);
    tf_shared_lock rlock(*(tensible_variable->GetRWLock()));
    Tensor in = ctx->input(1);
    Tensor default_value_tensor = ctx->input(2);
    OP_REQUIRES_ASYNC(ctx, default_value_tensor.shape().dims() == 0,
        errors::InvalidArgument("default_value should be scalar"), done);
    T default_value = default_value_tensor.scalar<T>()();
    const TensorShape& table_shape = tensible_variable->shape();
    const TensorShape& in_shape = in.shape();
    std::vector<int64> out_dims;
    int64 size, slice;
    TensorShape out_shape;
    BuildGatherShape(table_shape, in_shape, &out_shape, &slice, &size);
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, out_shape, &output), done);
    int64 max_id = tensible_variable->Size();
    int64* src = reinterpret_cast<int64*>(const_cast<char*>(
          in.tensor_data().data()));
    T* dst = reinterpret_cast<T*>(const_cast<char*>(
          output->tensor_data().data()));
    auto fn = [tensible_variable, max_id, slice, src, dst, default_value]
        (int64 offset, int64 size) -> Status {
      int64* psrc = src + offset;
      T* pdst = dst + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*psrc != HashTable::kNotAdmitted) {
          if (*psrc < 0 || *psrc >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *psrc);
          }
          memcpy(pdst, tensible_variable->GetSlice(*psrc), slice * sizeof(T));
        } else {
          for (int j = 0; j < slice; j++) {
            pdst[j] = default_value;
          }
        }
        psrc++;
        pdst += slice;
      }
      return Status::OK();
    };
    resource->Ref();
    auto done_fn = [done, ctx, resource](Status st) {
      resource->Unref();
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }
};

template <typename T, typename Functor>
class TensibleVariableSimpleUpdaterOp : public AsyncOpKernel {
 public:
  explicit TensibleVariableSimpleUpdaterOp(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    TensibleVariableResource* resource;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource), done);
    core::ScopedUnref s(resource);
    TensibleVariable* tensible_variable = resource->Internal();
    OP_REQUIRES_ASYNC(
        ctx, tensible_variable != nullptr,
        errors::FailedPrecondition("TensibleVariable is not initialized"), done);
    OP_REQUIRES_ASYNC(
        ctx, DataTypeToEnum<T>::value == tensible_variable->dtype(),
        errors::FailedPrecondition("TensibleVariable dtype mismatch"), done);
    tf_shared_lock rlock(*(tensible_variable->GetRWLock()));
    Tensor in = ctx->input(1);
    Tensor data = ctx->input(2);
    const TensorShape& table_shape = tensible_variable->shape();
    const TensorShape& in_shape = in.shape();
    std::vector<int64> out_dims;
    int64 size, slice;
    TensorShape out_shape;
    BuildGatherShape(table_shape, in_shape, &out_shape, &slice, &size);
    OP_REQUIRES_ASYNC(
        ctx, out_shape == data.shape(),
        errors::FailedPrecondition(
          "Data Shape Error ", out_shape.DebugString(),
          " vs ", data.shape().DebugString()), done);
    int64 max_id = tensible_variable->Size();
    int64* src = reinterpret_cast<int64*>(const_cast<char*>(
          in.tensor_data().data()));
    T* data_ptr = reinterpret_cast<T*>(const_cast<char*>(
          data.tensor_data().data()));
    auto fn = [tensible_variable, max_id, slice, src, data_ptr]
        (int64 offset, int64 size) -> Status {
      Functor functor;
      int64* psrc = src + offset;
      T* pdata = data_ptr + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*psrc != HashTable::kNotAdmitted) {
          if (*psrc < 0 || *psrc >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *psrc);
          }
          T* pvar = tensible_variable->GetSlice<T>(*psrc);
          for (int64 j = 0; j < slice; j++) {
            functor(pvar++, pdata++);
          }
        }
        psrc++;
      }
      return Status::OK();
    };
    resource->Ref();
    auto done_fn = [done, ctx, resource](Status st) {
      resource->Unref();
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }
};

template <typename T>
struct AssignFunctor {
  void operator()(T* x, T* y) {
    *x = *y;
  }
};

template <typename T>
struct AssignAddFunctor {
  void operator()(T* x, T* y) {
    *x += *y;
  }
};

template <typename T>
struct AssignSubFunctor {
  void operator()(T* x, T* y) {
    *x -= *y;
  }
};

template <typename T>
struct AssignMulFunctor {
  void operator()(T* x, T* y) {
    *x *= *y;
  }
};

template <typename T>
struct AssignDivFunctor {
  void operator()(T* x, T* y) {
    *x /= *y;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TensibleVariableOp").Device(DEVICE_CPU),
    ResourceHandleOp<TensibleVariableResource>);
REGISTER_KERNEL_BUILDER(
    Name("TensibleVariableInitializeOp").Device(DEVICE_CPU),
    TensibleVariableInitializeOp);
REGISTER_KERNEL_BUILDER(
    Name("TensibleVariableIsInitializedOp").Device(DEVICE_CPU),
    TensibleVariableIsInitializedOp);

#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("TensibleVariableGather")      \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          TensibleVariableGatherOp<type>);
TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("TensibleVariableScatterUpdate") \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<type>("dtype"),   \
                          TensibleVariableSimpleUpdaterOp<type, AssignFunctor<type>>);
TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("TensibleVariableScatterAdd")  \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          TensibleVariableSimpleUpdaterOp<type, AssignAddFunctor<type>>);
TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("TensibleVariableScatterSub")  \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          TensibleVariableSimpleUpdaterOp<type, AssignSubFunctor<type>>);
TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("TensibleVariableScatterMul")  \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          TensibleVariableSimpleUpdaterOp<type, AssignMulFunctor<type>>);
TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("TensibleVariableScatterDiv")  \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          TensibleVariableSimpleUpdaterOp<type, AssignDivFunctor<type>>);
TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow

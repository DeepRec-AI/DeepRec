/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "facade.h"
#include "tensor_buffer/embedding_buffer.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "embedding_variable.h"
#include <exception>
#include <vector>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

#if TF_VERSION_MAJOR == 2

template <typename Device>
class CreateVarOp : public OpKernel {
public:
    explicit CreateVarOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("trainable", &trainable_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_hashtable", &use_hashtable_));
        if (2 != shape_.dims()) {
            ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                "shape must be [vocabulary_size_per_gpu, embedding_vector_size]."));
            return;
        } 
        if (!shape_.IsFullyDefined()) {
            ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                "shape must be fully defined."));
            return;
        }
    }
    void Compute(OpKernelContext* ctx) override {
        const Tensor* var_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("var_name", &var_name_tensor));
        const Tensor* initial_value_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("initial_value", &initial_value_tensor)); 
        OP_REQUIRES(ctx, dtype_ == initial_value_tensor->dtype(), errors::Aborted(
                            __FILE__, ":", __LINE__, " The dtype is not consistent."));
        std::vector<int64_t> dims;
        auto helper = [this, &dims, &ctx](){
            dims.clear();
            for (auto iter = shape_.begin(); iter != shape_.end(); ++iter) {
                int64_t size_n = (*iter).size;
                if (size_n <= 0) {
                    ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                        "the dim ", size_n, " should be > 0."));
                    return;
                }
                dims.push_back(size_n);
            } // for iter
        };
        helper();
        const Tensor* local_replica_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("local_replica_id", &local_replica_id_tensor));

        std::string variable_name = var_name_tensor->flat<tstring>()(0);

        // generate unique variable name
        try {
            SparseOperationKit::Facade::instance()->generate_unique_name(trainable_, variable_name);
        } catch (const std::exception& error) {
            ctx->SetStatus(errors::Aborted("Error happens in generating unique variable name, due to ", error.what()));
            return;
        }

        // This is the handle for EmbeddingVariable
        Tensor* var_handle_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &var_handle_tensor));
        core::RefCountPtr<EmbeddingVariable> emb_variable;
        ResourceHandle emb_handle = MakeResourceHandle<EmbeddingVariable>(ctx, 
                                        /*container=*/"EmbeddingVariableContainer",
                                        /*name=*/variable_name);
        OP_REQUIRES_OK(ctx, LookupOrCreateResource<EmbeddingVariable>(ctx, emb_handle, &emb_variable,
                                        /*creator=*/[var_handle_tensor, &emb_handle](EmbeddingVariable** ptr){
                                            *ptr = new EmbeddingVariable(var_handle_tensor);
                                            (*ptr)->SetHandle(emb_handle);
                                            return Status::OK();
                                        }));
        Tensor tensor;
        // OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, shape_, &emb_tensor));
        Tensor* emb_tensor = &tensor;

        try {
            switch (dtype_) {
                case DT_FLOAT: {
                    // it is numpy value, used as initial_value
                    SparseOperationKit::Facade::instance()->create_variables(local_replica_id_tensor->scalar<int32_t>()(),
                                                                          initial_value_tensor->flat<float>().data(),
                                                                          use_hashtable_, dims, variable_name, 
                                                                          trainable_, emb_variable,
                                                                          emb_tensor);
                    break;
                } 
                case DT_STRING: {
                    // it specified the initializer
                    SparseOperationKit::Facade::instance()->create_variables(local_replica_id_tensor->scalar<int32_t>()(),
                                                                          initial_value_tensor->flat<tstring>()(0),
                                                                          use_hashtable_, dims, variable_name, 
                                                                          trainable_, emb_variable,
                                                                          emb_tensor);
                    break;
                }
                case DT_RESOURCE: {
                    // it is the initial_value and specifed memory space.
                    core::RefCountPtr<Var> init_variable;
                    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &init_variable));
                    Tensor* var_tensor = init_variable->tensor();
                    SparseOperationKit::Facade::instance()->create_variables(local_replica_id_tensor->scalar<int32_t>()(),
                                                                          var_tensor->flat<float>().data(),
                                                                          use_hashtable_, dims, variable_name, 
                                                                          trainable_, emb_variable,
                                                                          emb_tensor);
                    break;
                }
                default: {
                    ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                    "Not supported dtype for initial_value."));
                    return;
                }
            } // switch
        } catch (const std::exception& error) {
            ctx->SetStatus(errors::Aborted(error.what()));
            return;
        }

        // This is the handle for TF Var
        Tensor* handle_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &handle_tensor));
        core::RefCountPtr<Var> variable;
        dtype_and_shape_.dtype = dtype_;
        dtype_and_shape_.shape = shape_;
        ResourceHandle handle = MakeResourceHandle<Var>(ctx, 
                                        /*container=*/"VariableContainer",
                                        /*name=*/variable_name,
                                        /*DtypeAndPartialTensorShape=*/std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});
        OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(ctx, handle, &variable,
                                        /*creator=*/[&handle, &emb_tensor, handle_tensor](Var** ptr) {
                                            *ptr = new Var(DT_FLOAT);
                                            *(*ptr)->tensor() = *emb_tensor;
                                            (*ptr)->is_initialized = true;
                                            handle_tensor->scalar<ResourceHandle>()() = handle;
                                            return Status::OK();
                                        }));

        

        // set unique_var_name output
        Tensor* unique_var_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {}, &unique_var_name_tensor));
        unique_var_name_tensor->flat<tstring>()(0) = variable_name;
    }
private:
    bool trainable_;
    TensorShape shape_;
    DataType dtype_;
    bool use_hashtable_;
    DtypeAndPartialTensorShape dtype_and_shape_;
};

REGISTER_KERNEL_BUILDER(Name("CreateVar")
                        .Device(DEVICE_GPU)
                        .HostMemory("var_name")
                        .HostMemory("initial_value")
                        .HostMemory("local_replica_id")
                        .HostMemory("var_handle")
                        .HostMemory("handle")
                        .HostMemory("unique_var_name"),
                        CreateVarOp<GPUDevice>);

#else

template <typename Device>
class CreateVarOp : public OpKernel {
public:
    explicit CreateVarOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_and_shape_.dtype));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &dtype_and_shape_.shape));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
    }

    void Compute(OpKernelContext * ctx) override {
        if (var_name_ == ResourceHandle::ANONYMOUS_NAME) {
            AllocatorAttributes attr;
            attr.set_on_host(true);

            // create handle for EmbeddingVariable
            Tensor embedding_variable_handle;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), 
                                                   &embedding_variable_handle, attr));
            embedding_variable_handle.scalar<ResourceHandle>()() = 
                            MakeResourceHandle<EmbeddingVariable>(ctx,
                                /*container=*/"EmbeddingVariableContainer",
                                /*name=*/var_name_,
                                std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});
            ctx->set_output(0, embedding_variable_handle);

            // create handle for TF Var
            Tensor tf_variable_handle;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), 
                                                   &tf_variable_handle, attr));
            tf_variable_handle.scalar<ResourceHandle>()() = MakeResourceHandle<Var>(
                ctx, /*container=*/"VariableContainer", /*name=*/var_name_,
                std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});
            ctx->set_output(1, tf_variable_handle);
        } else {
            if (!initialized_.load()) {
                mutex_lock ml(mutex_);
                // Checking again to see if another thread has initialized the resource.
                if (!initialized_.load()) {
                    AllocatorAttributes attr;
                    attr.set_on_host(true);

                    // create handle for EmbeddingVariable
                    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                                           &embedding_variable_handle_, attr));
                    embedding_variable_handle_.scalar<ResourceHandle>()() = MakeResourceHandle<EmbeddingVariable>(
                                    ctx, /*container=*/"EmbeddingVariableContainer", 
                                    /*name=*/var_name_, 
                                    std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});

                    // create handle for TF Var
                    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                                            &tf_variable_handle_, attr));
                    tf_variable_handle_.scalar<ResourceHandle>()() = MakeResourceHandle<Var>(
                                    ctx, /*container=*/"VariableContainer",
                                    /*name=*/var_name_,
                                    std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});

                    initialized_.store(true);
                }
            }
            ctx->set_output(0, embedding_variable_handle_);
            ctx->set_output(1, tf_variable_handle_);
        }
    }
private:
    std::string var_name_;
    mutex mutex_;
    Tensor embedding_variable_handle_;
    Tensor tf_variable_handle_;
    std::atomic<bool> initialized_{false};
    DtypeAndPartialTensorShape dtype_and_shape_;
};


REGISTER_KERNEL_BUILDER(Name("CreateVar")
                        .Device(DEVICE_GPU)
                        .HostMemory("emb_var_handle")
                        .HostMemory("tf_var_handle"),
                        CreateVarOp<GPUDevice>);

#endif

} // namespace tensorflow
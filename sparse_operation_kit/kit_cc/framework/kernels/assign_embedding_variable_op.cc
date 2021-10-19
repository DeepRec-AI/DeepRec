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

#if TF_VERSION_MAJOR == 1

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

template <typename Device>
class AssignEmbeddingVariableOp : public OpKernel {
public:
    explicit AssignEmbeddingVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("trainable", &trainable_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_hashtable", &use_hashtable_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_and_shape_.dtype));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &dtype_and_shape_.shape));
        if (2 != dtype_and_shape_.shape.dims()) {
            ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                "shape must be [vocabulary_size_per_gpu, embedding_vector_size]."));
            return;
        }
        if (!dtype_and_shape_.shape.IsFullyDefined()) {
            ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                "shape must be fully defined."));
            return;
        }
        shape_convertor(ctx);
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* var_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("var_name", &var_name_tensor));
        const Tensor* initial_value_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("initial_value", &initial_value_tensor));
        OP_REQUIRES(ctx, initial_value_tensor->dtype() == DT_STRING, errors::Aborted(
                    __FILE__, ":", __LINE__, " Currently, only string can be used to"
                    " set initializer."));
        const Tensor* local_replica_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("local_replica_id", &local_replica_id_tensor));

        std::string variable_name = var_name_tensor->flat<tstring>()(0);
        // generate unique variable name
        try {
            SparseOperationKit::Facade::instance()->generate_unique_name(trainable_, variable_name);
        } catch (const std::exception& error) {
            ctx->SetStatus(errors::Aborted("Error happens when generating unique name, "
                                           "due to ", error.what()));
            return;
        }
        OP_REQUIRES(ctx, var_name_tensor->flat<tstring>()(0) == variable_name, 
                    errors::Aborted(__FILE__, ":", __LINE__, " there already exist ", 
                    var_name_tensor->flat<tstring>()(0)));

        // Create resource for EmbeddingVariable
        core::RefCountPtr<EmbeddingVariable> emb_variable;
        OP_REQUIRES_OK(ctx, LookupOrCreateResource<EmbeddingVariable>(ctx, HandleFromInput(ctx, 0), &emb_variable,
                                    /*creator=*/[](EmbeddingVariable** ptr){
                                        *ptr = new EmbeddingVariable();
                                        return Status::OK();
                                    }));
        Tensor tensor; // used to hold the pointer to allocated GPU memory
        try {
            SparseOperationKit::Facade::instance()->create_variables(local_replica_id_tensor->scalar<int32_t>()(),
                                                                     initial_value_tensor->flat<tstring>()(0),
                                                                     use_hashtable_, dims_, variable_name, 
                                                                     trainable_, emb_variable, &tensor);
        } catch (const std::exception& error) {
            ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " errors happens due to ", error.what()));
            return;
        }

        // create resource for TF Var
        core::RefCountPtr<Var> tf_variable;
        OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(ctx, HandleFromInput(ctx, 1), &tf_variable,
                                    /*creator=*/[&tensor](Var** ptr){
                                        *ptr = new Var(DT_FLOAT);
                                        *(*ptr)->tensor() = tensor;
                                        (*ptr)->is_initialized = true;
                                        return Status::OK();
                                    }));
    }
private:
    bool trainable_;
    DtypeAndPartialTensorShape dtype_and_shape_;
    std::vector<int64_t> dims_;
    bool use_hashtable_;

    void shape_convertor(OpKernelConstruction* ctx) {
        dims_.clear();
        const auto& shape = dtype_and_shape_.shape;
        for (auto iter = shape.begin(); iter != shape.end(); ++iter) {
            int64_t size_n = (*iter).size;
            if (size_n <= 0) {
                ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                        "The dim ", size_n, " should be > 0."));
                return;
            } 
            dims_.push_back(size_n);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AssignEmbeddingVariable")
                        .Device(DEVICE_GPU)
                        .HostMemory("emb_var_handle")
                        .HostMemory("tf_var_handle")
                        .HostMemory("var_name")
                        .HostMemory("initial_value")
                        .HostMemory("local_replica_id"),
                        AssignEmbeddingVariableOp<GPUDevice>);

} // namespace tensorflow

#endif
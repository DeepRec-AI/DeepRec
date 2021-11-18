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

#ifndef EMBEDDING_VARIABLE_H
#define EMBEDDING_VARIABLE_H

#include "parameters/param_interface.h"
#include "tensorflow/core/framework/resource_var.h"

namespace tensorflow { 

class EmbeddingVariable : public ResourceBase {
public:
#if TF_VERSION_MAJOR == 2
    EmbeddingVariable(Tensor* handle_tensor);
    void SetHandle(ResourceHandle& handle);
#else 
    EmbeddingVariable();
#endif
    ~EmbeddingVariable();
    std::string DebugString() const override;
    Tensor* tensor();
    void set_param(const std::shared_ptr<SparseOperationKit::ParamInterface>& param);
    void get_param(std::shared_ptr<SparseOperationKit::ParamInterface>& param);
    mutex* mu();
private:
    Tensor* handle_tensor_ = nullptr;
    std::shared_ptr<SparseOperationKit::ParamInterface> param_;
    mutex mu_;
};

} // namespace tensorflow

#endif
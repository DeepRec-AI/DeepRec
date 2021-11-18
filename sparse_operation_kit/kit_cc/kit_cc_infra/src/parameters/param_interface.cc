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

#include "parameters/param_interface.h"
#include "common.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace SparseOperationKit {

void ParamInterface::set_user(std::shared_ptr<EmbeddingLayer>& embedding) {
    // It is not compulsory for the subclass to override this function.
    throw std::runtime_error(ErrorBase + "Not implemented.");
}

void ParamInterface::let_user_dump_to_file(const std::string filepath) {
    // by default, it does nothing.
}

void ParamInterface::let_user_restore_from_file(const std::string filepath) {
    // by default, it does nothing.
}

void ParamInterface::let_user_load_embedding_values(const std::vector<std::shared_ptr<Tensor>> &tensor_list) {
    // by default, it does nothing
}

std::shared_ptr<Tensor>& ParamInterface::get_tensor(const size_t local_replica_id) {
    return get_embedding_table_tensor(local_replica_id);
}

} // namespace SparseOperationKit
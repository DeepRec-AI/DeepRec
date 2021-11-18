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

#ifndef OPERATION_HELPER_H
#define OPERATION_HELPER_H

#include "operation/builder_container.h"
#include "dispatcher/dispatcher_builder.h"
#include "embeddings/embedding_lookuper_builder.h"
#include "common.h"
#include <string>
#include <memory>

namespace SparseOperationKit {

template <typename DispatcherClass>
int register_input_builder_helper(const std::string dispatcher_name) {
    auto temp = std::shared_ptr<Builder>(new InputDispatcherBuilder<DispatcherClass>());
    InputContainer::instance("input_dispatcher_builders")->push_back(dispatcher_name, temp);
    return 0;
}

template <typename OperationClass>
int register_operation_builder_helper(const std::string operation_name) {
    auto temp = std::shared_ptr<Builder>(new OperationBuilder<OperationClass>());
    OperationContainer::instance("operation_builders")->push_back(operation_name, temp);
    return 0;
}

template <typename DispatcherClass>
int register_output_builder_helper(const std::string dispatcher_name) {
    auto temp = std::shared_ptr<Builder>(new OutputDispatcherBuilder<DispatcherClass>());
    OutputContainer::instance("output_dispatcher_builders")->push_back(dispatcher_name, temp);
    return 0;
}

template <typename EmbeddingLookuperClass>
int register_emb_lookuper_helper(const std::string lookuper_name) {
    auto temp = std::shared_ptr<Builder>(new EmbeddingLookuperBuilder<EmbeddingLookuperClass>());
    LookuperContainer::instance("embedding_lookuper_builders")->push_back(lookuper_name, temp);
    return 0;
}

#define REGISTER_INPUT_DISPATCHER_BUILDER(dispatcher_name, dispatcher_class)  \
    auto _##dispatcher_class = register_input_builder_helper<dispatcher_class>(dispatcher_name); 

#define REGISTER_OPERATION_BUILDER(operation_name, operation_class) \
    auto _##operation_class = register_operation_builder_helper<operation_class>(operation_name);

#define REGISTER_OUTPUT_DISPATHER_BUILDER(dispatcher_name, dispatcher_class) \
    auto _##dispatcher_class = register_output_builder_helper<dispatcher_class>(dispatcher_name);

#define REGISTER_EMB_LOOKUPER_BUILDER(lookuper_name, lookuper_class) \
    auto _##lookuper_class = register_emb_lookuper_helper<lookuper_class>(lookuper_name);

} // namespace SparseOperationKit

#endif // OPERATION_HELPER_H
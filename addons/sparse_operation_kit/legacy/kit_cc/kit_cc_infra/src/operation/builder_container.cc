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

#include "operation/builder_container.h"

#include "common.h"

namespace SparseOperationKit {

BuilderContainer::BuilderContainer(const std::string name) : name_(name) {}

void BuilderContainer::push_back(const std::string builder_name,
                                 const std::shared_ptr<Builder> builder) {
  auto iter = components_.find(builder_name);
  if (components_.end() != iter)
    throw std::runtime_error(ErrorBase + "There exists a builder whose name is " + builder_name +
                             " in container: " + name_);

  components_.emplace(std::make_pair(builder_name, builder));
}

std::shared_ptr<Builder> BuilderContainer::get_builder(const std::string builder_name) {
  auto iter = components_.find(builder_name);
  if (components_.end() == iter)
    throw std::runtime_error(ErrorBase + "Cannot find " + builder_name + " in container: " + name_);

  return iter->second;
}

std::vector<std::string> BuilderContainer::get_builder_names() const {
  std::vector<std::string> builder_names;
  for (auto iter : components_) {
    builder_names.emplace_back(iter.first);
  }
  return builder_names;
}

InputContainer::InputContainer(const std::string name) : BuilderContainer(name) {}

InputContainer* InputContainer::instance(const std::string name) {
  static InputContainer instance(name);
  return &instance;
}

OutputContainer::OutputContainer(const std::string name) : BuilderContainer(name) {}

OutputContainer* OutputContainer::instance(const std::string name) {
  static OutputContainer instance(name);
  return &instance;
}

OperationContainer::OperationContainer(const std::string name) : BuilderContainer(name) {}

OperationContainer* OperationContainer::instance(const std::string name) {
  static OperationContainer instance(name);
  return &instance;
}

LookuperContainer::LookuperContainer(const std::string name) : BuilderContainer(name) {}

LookuperContainer* LookuperContainer::instance(const std::string name) {
  static LookuperContainer instance(name);
  return &instance;
}

}  // namespace SparseOperationKit
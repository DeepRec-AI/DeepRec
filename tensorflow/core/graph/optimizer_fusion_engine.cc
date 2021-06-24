/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include "tensorflow/core/graph/optimizer_fusion_engine.h"
#include "tensorflow/core/graph/optimizer_fusion_engine_impl.h"
#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

bool OptimizeFusion(Graph* g) {
  bool changed = false;
  std::vector<std::unique_ptr<TemplateBase>> templates;

  for (auto& t : templates) {
    std::unique_ptr<OptimizerFusionImpl> opt(new OptimizerFusionImpl(g, t.get()));
    changed |= opt->Optimize();
  }
  return changed;
}

}  // namespace tensorflow

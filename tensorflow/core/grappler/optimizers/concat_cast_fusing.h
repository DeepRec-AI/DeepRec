/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONCAT_CAST_FUSING_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONCAT_CAST_FUSING_H_

#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

// ConcatCastFusing optimization for a graph.
class ConcatCastFusing : public GraphOptimizer {
    public:

    explicit ConcatCastFusing(DeviceBase* cpu_device);
    ConcatCastFusing(RewriterConfig::Toggle opt_level, DeviceBase* cpu_device);

    ~ConcatCastFusing() override {}

    string name() const override { return "concat_cast_fusing"; };

    bool UsesFunctionLibrary() const override { return false; }

    Status Optimize(Cluster* cluster, const GrapplerItem& item,
                    GraphDef* output) override;

    void Feedback(Cluster* cluster, const GrapplerItem& item,
                  const GraphDef& optimize_output, double result) override;

    RewriterConfig::Toggle opt_level_;
    DeviceBase* cpu_device_;
    std::unique_ptr<DeviceBase> owned_device_;

    std::unique_ptr<ResourceMgr> resource_mgr_;
    GraphDef* graph_;
    std::unique_ptr<NodeMap> node_map_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONCAT_CAST_FUSING_H_

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MKL_LAYOUT_PASS_LISTS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MKL_LAYOUT_PASS_LISTS_H_

#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

class MklLayoutPassLists {
 public:
  MklLayoutPassLists() {}

  static gtl::FlatSet<string> FinalList() {
    auto list = gtl::FlatSet<string>{
      "MatMul",
      "_FusedMatMul",
      "BatchMatMul",
      "BatchMatMulV2",
      "BiasAdd",
      "BiasAddGrad",
      "_FusedBatchMatMul",
      "_FusedBatchMatMulV2",
      "Identity",
      "LeakyRelu",
      "LeakyReluGrad",
      "Relu",
      "ReluGrad",
      "Relu6",
      "Relu6Grad",
      "Gelu",
      "GeluGrad",
      "Tanh",
      "TanhGrad",
      "Reshape"
    };

    UpdateList(&list);
    return list;
  }

  static bool FindFusedMatMul() {
    return FinalList().count("_FusedMatMul");
  }

 protected:
  // Adds or removes ops from list if certain environmental variables are set.
  static void UpdateList(gtl::FlatSet<string>* list) {
    bool enable_reco_ops_list;

    // Enable "TF_MKL_PRIMITIVE_ONLY_FOR_RECO" by default
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_MKL_PRIMITIVE_ONLY_FOR_RECO",
                                /*default_val=*/true, &enable_reco_ops_list));
    VLOG(1) << "MklLayoutRewritePass: TF_MKL_PRIMITIVE_ONLY_FOR_RECO = "
            << enable_reco_ops_list;

    if (!enable_reco_ops_list){
      list->clear();
      return;
    }
    string add_env_var =
        "TF_MKL_LAYOUT_PASS_GRAPH_REWRITE_LIST_ADD";
    string remove_env_var =
        "TF_MKL_LAYOUT_PASS_GRAPH_REWRITE_LIST_REMOVE";
    string to_add, to_remove;
    TF_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
    TF_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
    for (const auto& x : str_util::Split(to_add, ",")) {
      list->insert(x);
    }
    for (const auto& x : str_util::Split(to_remove, ",")) {
      list->erase(x);
    }
  }
};
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MKL_LAYOUT_PASS_LISTS_H_

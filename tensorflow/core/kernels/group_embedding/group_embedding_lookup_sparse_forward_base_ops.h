/* Copyright 2022 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/unique_ali_op_util.h"

namespace tensorflow {
// It's suggested that all CPU GroupEmbedding operations inherit from this base class.
template <typename TKey, typename TValue>
class GroupLookupBaseCpuOp : public OpKernel {
 public:
  explicit GroupLookupBaseCpuOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &m_combiner));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &m_num_lookup));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &m_dimension));
    // OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(c, c->GetAttr("ignore_weights", &m_ignore_weights));
    OP_REQUIRES_OK(c, c->GetAttr("is_sequence", &m_is_sequence));
    OP_REQUIRES_OK(c, ReadInt64FromEnvVar(kUniqueOpPartitionSizeEnv,
                                          kPartitionSize, &partition_size_));
    OP_REQUIRES(
        c, partition_size_ > 0,
        errors::InvalidArgument("Invaild PARTITION_SIZE=", partition_size_));
    OP_REQUIRES_OK(c, ReadBoolFromEnvVar(kUniqueOpSerialEnv, false, &serial_));
    OP_REQUIRES_OK(
        c, ReadInt64FromEnvVar(kUniqueOpUniqRatioHint, kDefaultUniqueRatioHint,
                               &unique_ratio_hint_));
    OP_REQUIRES(c, unique_ratio_hint_ > 0,
                errors::InvalidArgument("Invaild ", kUniqueOpUniqRatioHint, "=",
                                        unique_ratio_hint_));
  }

 protected:
  // float max_norm_;
  int m_num_lookup;
  int m_dimension;
  bool m_is_use_default_value_tensor;
  bool m_ignore_weights;
  bool m_is_sequence;
  std::string m_combiner;
  bool serial_ = false;
  int64 partition_size_ = 0;
  int64 unique_ratio_hint_;
  UniqueMaps map_flag_ = GOOGLE;  // "GOOGLE" dense hash map is default
  const int64 kDefaultUniqueRatioHint = 4;
  const char* kUniqueOpSerialEnv = "DEEPREC_UNIQUE_OP_SERIAL";
  const char* kUniqueOpUniqRatioHint = "DEEPREC_UNIQUE_OP_UNIQ_RATIO_HINT";
  const char* kUniqueOpPartitionSizeEnv = "DEEPREC_UNIQUE_OP_PARTITION_SIZE";
};

}  // namespace tensorflow
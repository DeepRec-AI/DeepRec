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

// Algorithms and data structures for partition assignment.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PARTITION_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PARTITION_ASSIGNMENT_H_

#include <iosfwd>
#include <map>
#include <memory>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

// Encapsulates the launch dimensions of a kernel, e.g., the block count and the
// number of threads per block.
class LaunchDimensions {
 public:
  struct Dim3D {
    int64 x, y, z;
  };

  // The default constructor creates a launch dimension that indicate
  // single-threaded execution.
  LaunchDimensions()
      : block_counts_({1, 1, 1}), thread_counts_per_block_({1, 1, 1}) {}

  LaunchDimensions(int64 block_x_count, int64 thread_x_count_per_block)
      : block_counts_({block_x_count, 1, 1}),
        thread_counts_per_block_({thread_x_count_per_block, 1, 1}) {}

  LaunchDimensions(const Dim3D& block_counts,
                   const Dim3D& thread_counts_per_block)
      : block_counts_(block_counts),
        thread_counts_per_block_(thread_counts_per_block) {}

  Dim3D block_counts() const { return block_counts_; }

  Dim3D thread_counts_per_block() const { return thread_counts_per_block_; }

  int64 launch_bound() const {
    return block_counts_.x * thread_counts_per_block_.x * block_counts_.y *
           thread_counts_per_block_.y * block_counts_.z *
           thread_counts_per_block_.z;
  }

 private:
  Dim3D block_counts_;
  Dim3D thread_counts_per_block_;
};

std::ostream& operator<<(std::ostream& out,
                         const LaunchDimensions& launch_dims);

// Returns the maximum number of threads per block allowed by the device.
int64 ThreadsPerBlockLimit(const se::DeviceDescription& device_desc);

LaunchDimensions CalculateLaunchDimensions(
    const Shape& shape, const se::DeviceDescription& device_desc,
    int unroll_factor = 1, bool few_waves = false);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PARTITION_ASSIGNMENT_H_

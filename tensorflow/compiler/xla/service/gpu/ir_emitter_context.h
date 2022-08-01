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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_

#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"

namespace xla {
namespace gpu {
// IrEmitterContext encapsulates common (mutable and immutable) data structures
// used by both IrEmitterNested and IrEmitterUnnested, such as the buffer
// assignment and the name uniquer.
class IrEmitterContext {
 public:
  // cuda_compute_capability is nullopt if we're not compiling for NVIDIA GPUs.
  IrEmitterContext(
      const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
      std::string platform_name, GpuDeviceInfo gpu_device_info,
      absl::optional<CudaComputeCapability> cuda_compute_capability,
      llvm::Module* llvm_module)
      : hlo_module_(hlo_module),
        buffer_assignment_(buffer_assignment),
        platform_name_(std::move(platform_name)),
        gpu_device_info_(gpu_device_info),
        cuda_compute_capability_(cuda_compute_capability),
        llvm_module_(llvm_module) {}
  // Disallow copy and assign.
  IrEmitterContext(const IrEmitterContext&) = delete;
  IrEmitterContext& operator=(const IrEmitterContext&) = delete;

  // Simple accessors.
  const HloModule& hlo_module() const { return *hlo_module_; }
  const BufferAssignment& buffer_assignment() const {
    return *buffer_assignment_;
  }
  absl::string_view platform_name() const { return platform_name_; }
  GpuDeviceInfo gpu_device_info() const { return gpu_device_info_; }
  absl::optional<CudaComputeCapability> cuda_compute_capability() const {
    return cuda_compute_capability_;
  }
  llvm::Module* llvm_module() { return llvm_module_; }
  NameUniquer* name_uniquer() { return &name_uniquer_; }

 private:
  const HloModule* hlo_module_;
  const BufferAssignment* buffer_assignment_;
  std::string platform_name_;
  GpuDeviceInfo gpu_device_info_;
  absl::optional<CudaComputeCapability> cuda_compute_capability_;
  llvm::Module* llvm_module_;
  NameUniquer name_uniquer_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_

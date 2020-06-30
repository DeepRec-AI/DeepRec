#include "tensorflow/stream_executor/cuda/ptxas_utils.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_

namespace stream_executor {
static port::StatusOr<std::vector<uint8>> CompileGpuAsm(
    int device_ordinal,
    const char* ptx_contents,
    stream_executor::gpu::GpuAsmOpts options) {
  return CompilePtx(device_ordinal, ptx_contents, options);
}

static port::StatusOr<absl::Span<const uint8>> CompileGpuAsmOrGetCached(
    int device_ordinal, const char* ptx,
    stream_executor::gpu::GpuAsmOpts compilation_options) {
  return CompilePtxOrGetCached(device_ordinal, ptx, compilation_options);
}

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_

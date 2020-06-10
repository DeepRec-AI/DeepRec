#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_

namespace stream_executor {

  port::StatusOr<std::vector<uint8>> CompileGpuAsm(int device_ordinal,
                                                 const char* ptx_contents,
                                                 GpuAsmOpts options) {
  return CompilePtx(device_ordinal, ptx_contents, options);
}

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_

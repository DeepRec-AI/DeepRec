#include "third_party/gpus/cuda/include/cublasLt.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"

// Implements the cuBLASLt API by forwarding to cuBLASLt loaded from the DSO.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void* GetDsoHandle() {
#ifdef PLATFORM_GOOGLE
  return nullptr;
#else
  static auto handle = []() -> void* {
    auto handle_or =
        stream_executor::internal::DsoLoader::GetCublasLtDsoHandle();
    if (!handle_or.ok()) return nullptr;
    return handle_or.ValueOrDie();
  }();
  return handle;
#endif
}

template <typename T>
T LoadSymbol(const char* symbol_name) {
  void* symbol = nullptr;
  if (auto handle = GetDsoHandle()) {
    stream_executor::port::Env::Default()
        ->GetSymbolFromLibrary(handle, symbol_name, &symbol)
        .IgnoreError();
  }
  return reinterpret_cast<T>(symbol);
}

void LogFatalSymbolNotFound(const char* symbol_name) {
  LOG(FATAL) << symbol_name << " symbol not found.";
}

cublasStatus_t GetSymbolNotFoundError() { return CUBLAS_STATUS_INTERNAL_ERROR; }
}  // namespace

// We only use cublasLt from CUDA 11.0 onward.
#if CUDA_VERSION >= 11000
#include "tensorflow/stream_executor/cuda/cublasLt_11_0.inc"
#endif

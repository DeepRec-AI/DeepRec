/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "third_party/tensorrt/NvInfer.h"

// Implements the TensorRT API by forwarding to TensorRT loaded from the DSO.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void* GetDsoHandle() {
#ifdef PLATFORM_GOOGLE
  return nullptr;
#else
  static auto handle = []() -> void* {
    auto handle_or =
        stream_executor::internal::DsoLoader::GetNvInferDsoHandle();
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
    tensorflow::Env::Default()
        ->GetSymbolFromLibrary(handle, symbol_name, &symbol)
        .IgnoreError();
  }
  return reinterpret_cast<T>(symbol);
}

void LogFatalSymbolNotFound(const char* symbol_name) {
  LOG(FATAL) << symbol_name << " symbol not found.";
}
}  // namespace

#if IS_TRT_VERSION_GE(8, 0, 0, 0)
#include "tensorflow/compiler/tf2tensorrt/stub/NvInfer_8_0.inc"
#elif IS_TRT_VERSION_GE(5, 1, 0, 0)
#include "tensorflow/compiler/tf2tensorrt/stub/NvInfer_5_1.inc"
#elif IS_TRT_VERSION_GE(5, 0, 0, 0)
#include "tensorflow/compiler/tf2tensorrt/stub/NvInfer_5_0.inc"
#else
#error TensorRT version earlier than 5 is not supported.
#endif


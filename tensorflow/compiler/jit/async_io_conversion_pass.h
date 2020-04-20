/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_ASYNC_IO_CONVERSION_PASS_
#define TENSORFLOW_COMPILER_JIT_ASYNC_IO_CONVERSION_PASS_

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// This pass converts synchronous XLA outputs (i.e., Retval) to asynchronous
// _XlaAsyncOutSend and _XlaAsyncOutRecv. Unlike Retvals, the _XlaAsyncOutSends
// can output tensors upon availability of data items and send them to the
// paired _XlaAsyncOutRecvs before the XLA clusters finish execution. As such,
// the consuming operations of the XLA clusters can start executing earlier
// without waiting for the XLA clusters to finish.
//
// TODO: extend to the inputs of XLA clusters.
class AsyncIoConversionPass : public GraphOptimizationPass {
 public:
  explicit AsyncIoConversionPass(
      absl::optional<int> async_io_level = absl::nullopt)
      : async_io_level_(async_io_level) {}

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  // If 1, replaces output edges of XLA clusters with pairs of _XlaAsyncOutSend
  // and _XlaAsyncOutRecv for asynchronous outputs based on heuristics. The
  // current heuristic replaces only data edges whose destination nodes are
  // HorovodAllReduces (for details, see IsConvertibleAndProfitable in
  // async_io_conversion_pass.cc). If 2, replaces all output data edges with
  // async outputs whenever legal. Off if other values.
  absl::optional<int> async_io_level_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_ASYNC_IO_CONVERSION_PASS_

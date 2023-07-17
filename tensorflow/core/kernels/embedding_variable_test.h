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
======================================================================*/
#ifndef TENSORFLOW_CORE_KERNELS_EMBEDING_VARIABLE_TEST_H
#define TENSORFLOW_CORE_KERNELS_EMBEDING_VARIABLE_TEST_H
#include <thread>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#endif //GOOGLE_CUDA

#include <time.h>
#include <sys/resource.h>
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#ifdef TENSORFLOW_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif

namespace tensorflow {
namespace embedding {
struct ProcMemory {
  long size;      // total program size
  long resident;  // resident set size
  long share;     // shared pages
  long trs;       // text (code)
  long lrs;       // library
  long drs;       // data/stack
  long dt;        // dirty pages

  ProcMemory() : size(0), resident(0), share(0),
                 trs(0), lrs(0), drs(0), dt(0) {}
};

ProcMemory getProcMemory() {
  ProcMemory m;
  FILE* fp = fopen("/proc/self/statm", "r");
  if (fp == NULL) {
    LOG(ERROR) << "Fail to open /proc/self/statm.";
    return m;
  }

  if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
             &m.size, &m.resident, &m.share,
             &m.trs, &m.lrs, &m.drs, &m.dt) != 7) {
    fclose(fp);
    LOG(ERROR) << "Fail to fscanf /proc/self/statm.";
    return m;
  }
  fclose(fp);

  return m;
}

double getSize() {
  ProcMemory m = getProcMemory();
  return m.size;
}

double getResident() {
  ProcMemory m = getProcMemory();
  return m.resident;
}

EmbeddingVar<int64, float>* CreateEmbeddingVar(
    int value_size, Tensor& default_value,
    int64 default_value_dim, int64 filter_freq = 0,
    int64 steps_to_live = 0,
    float l2_weight_threshold=-1.0) {
  std::string layout_type = "light";
  if (filter_freq != 0) {
    layout_type = "normal";
  }

  if (steps_to_live != 0) {
    if (layout_type == "light") {
      layout_type = "normal_contiguous";
    }
  }
  auto embedding_config = EmbeddingConfig(
			0, 0, 1, 0, "emb_var", steps_to_live,
			filter_freq, 999999, l2_weight_threshold, layout_type,
			0, -1.0, DT_UINT64, default_value_dim,
			0.0, false, false, false);
  auto storage =
      embedding::StorageFactory::Create<int64, float>(
          embedding::StorageConfig(
              embedding::StorageType::DRAM, "",
              {1024, 1024, 1024, 1024}, layout_type,
              embedding_config),
          cpu_allocator(),
          "emb_var");
	auto ev = new EmbeddingVar<int64, float>(
      "emb_var",
      storage,
      embedding_config,
      cpu_allocator());
	ev->Init(default_value, default_value_dim);
  return ev;
}
} //namespace embedding
} //namespace tensorflow
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_FACTORY_H_

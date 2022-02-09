#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_PARTITION_SELECT_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_PARTITION_SELECT_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace fused_embedding {

enum PartitionStrategy { DIV };

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_PARTITION_SELECT_H_
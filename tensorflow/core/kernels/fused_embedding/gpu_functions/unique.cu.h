#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_UNIQUE_CU_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_UNIQUE_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace fused_embedding {

// Not really a TensorFlow Op and output "idx_of_input_to_unique" is
// not the same as output "idx" in TensorFlow 2.x's UniqueWithCounts operator
template <typename T, typename TIndex>
void UniqueWithCountsGPU(OpKernelContext* context, const Tensor& input,
                         Tensor* unique_keys_out,
                         Tensor* idx_of_input_to_unique,
                         Tensor* unique_counts_out);

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_UNIQUE_CU_H_
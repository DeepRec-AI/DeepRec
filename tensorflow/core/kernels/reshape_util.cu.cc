/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/reshape_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/framework/typed_allocator.h"

namespace tensorflow {

#define ERRNO_ONE_DIMENSION_BE_NEG_ONE  -1
#define ERRNO_TARGET_SHAPE_NO_NEG -2
#define ERRNO_SHAPE_INFERENCE -3
#define ERRNO_INFERENCE_NUM_MISMATCH -4
#define ERRNO_NUM_MISMATCH -5

__global__ __launch_bounds__(1024) void ReshapeGPUKernel(
        const int64 *input_indices_in, int64 *result_indices,
        const int64 *input_strides, const int64 *output_strides,
        const int64 input_rank, const int64 output_rank, const int64 nnz) {
    GPU_1D_KERNEL_LOOP(i, nnz) {
        int64 id = 0;
        for (int j = 0; j < input_rank; ++j) {
            id += input_indices_in[i*input_rank+j] * input_strides[j];
        }
        for (int j = 0; j < output_rank; ++j) {
            result_indices[i*output_rank+j] = id / output_strides[j];
            id %= output_strides[j];
        }
    }
}

void ReshapeGPU(OpKernelContext *context, const Tensor &input_indices_in,
             const Tensor &input_shape_in, const Tensor &target_shape_in,
             int output_indices_idx, int output_shape_idx) {
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices_in.shape()),
              errors::InvalidArgument(
                  "Input indices should be a matrix but received shape ",
                  input_indices_in.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
              errors::InvalidArgument(
                  "Input shape should be a vector but received shape ",
                  input_shape_in.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(target_shape_in.shape()),
              errors::InvalidArgument(
                  "Target shape should be a vector but received shape ",
                  target_shape_in.shape().DebugString()));

  const int64 input_rank = input_shape_in.NumElements();
  const int64 output_rank = target_shape_in.NumElements();
  const TensorShape input_shape(input_shape_in.vec<int64>());
  const int64 dense_size = input_shape.num_elements();
  const int64 nnz = input_indices_in.shape().dim_size(0);

  // Compute the output shape. Determine product of specified dimensions, and
  // find the index of the unspecified one.
  TensorShape output_shape;
  int64 product = 1;
  int unknown_index = -1;
  auto target_shape = target_shape_in.vec<int64>();
  for (int d = 0; d < output_rank; ++d) {
    const int64 size = target_shape(d);
    if (size == -1) {
      OP_REQUIRES(
          context, unknown_index == -1,
          errors::InvalidArgument("only one output dimension may be -1, "
                                  "not both ",
                                  unknown_index, " and ", d));
      unknown_index = d;
      output_shape.AddDim(1);
    } else {
      OP_REQUIRES(context, size >= 0,
                  errors::InvalidArgument("size ", d,
                                          " must be non-negative, not ", size));
      product *= size;
      output_shape.AddDim(size);
    }
  }
  if (unknown_index != -1) {
    OP_REQUIRES(
        context, product > 0,
        errors::InvalidArgument("reshape cannot infer the missing "
                                "input size for an empty tensor unless all "
                                "specified input sizes are non-zero"));
    const int64 missing = dense_size / product;
    OP_REQUIRES(
        context, product * missing == dense_size,
        errors::InvalidArgument(
            "Inferenced element num of output SparseTensor "
            "is different from in SparseTensor. "
            "Input to reshape is a SparseTensor with ", dense_size,
            " dense values, but the requested shape requires a multiple of ",
            product, ". input_shape=", input_shape.DebugString(),
            " output_shape=", output_shape.DebugString()));
    output_shape.set_dim(unknown_index, missing);
  }

  OP_REQUIRES(
      context, output_shape.num_elements() == dense_size,
      errors::InvalidArgument("Element num of input SparseTensor is different "
                              "from output SparseTensor. "
                              "Input to reshape is a tensor with ", dense_size,
                              " dense values, but the requested shape has ",
                              output_shape.num_elements(),
                              ". input_shape=", input_shape.DebugString(),
                              " output_shape=", output_shape.DebugString()));
  Tensor *result_shape = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(output_shape_idx,
                                                   TensorShape({output_rank}),
                                                   &result_shape));
  auto output_shape_vec = result_shape->vec<int64>();
  gtl::InlinedVector<int64, 8> h_output_shape(output_rank);
  for (int j = 0; j < output_shape.dims(); ++j) {
    h_output_shape[j] = output_shape.dim_size(j);
  }
  cudaMemcpy(output_shape_vec.data(), h_output_shape.data(), 
      output_shape.dims()*sizeof(int64), cudaMemcpyHostToDevice);
  // Optimize for reshaping to the same shape.
  if (input_shape == output_shape) {
    context->set_output(output_indices_idx, input_indices_in);
    return;
  }

  Tensor *result_indices = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(output_indices_idx,
                                          TensorShape({nnz, output_rank}),
                                          &result_indices));

  if (nnz == 0) return;

  auto *allocator = context->get_allocator(AllocatorAttributes());

  gtl::InlinedVector<int64, 8> input_strides(input_rank);
  int64 *d_input_strides = nullptr;
  if (input_rank > 0) {
    input_strides[input_rank - 1] = 1;
    for (int d = input_rank - 2; d >= 0; --d) {
      input_strides[d] = input_strides[d + 1] * input_shape.dim_size(d + 1);
    }
    d_input_strides = TypedAllocator::Allocate<int64>(allocator, input_rank, AllocationAttributes());
    cudaMemcpy(d_input_strides, input_strides.data(), input_rank*sizeof(int64),
               cudaMemcpyHostToDevice);
  }

  gtl::InlinedVector<int64, 8> output_strides(output_rank);
  int64 *d_output_strides = nullptr;
  if (output_rank > 0) {
    output_strides[output_rank - 1] = 1;
    for (int d = output_rank - 2; d >= 0; --d) {
      output_strides[d] = output_strides[d + 1] * output_shape.dim_size(d + 1);
    }
    d_output_strides = TypedAllocator::Allocate<int64>(allocator, output_rank, AllocationAttributes());
    cudaMemcpy(d_output_strides, output_strides.data(), output_rank*sizeof(int64),
               cudaMemcpyHostToDevice);
  }

  auto input_ind = input_indices_in.matrix<int64>();
  auto output_ind = result_indices->matrix<int64>();

  const Eigen::GpuDevice& device = context->template eigen_device<Eigen::GpuDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(nnz, device);
  GpuLaunchKernel(ReshapeGPUKernel, config.block_count, config.thread_per_block, 
                0, device.stream(), 
                input_indices_in.matrix<int64>().data(), 
                result_indices->matrix<int64>().data(),
                d_input_strides, d_output_strides,
                input_rank, output_rank, nnz);

  TypedAllocator::Deallocate(allocator, d_input_strides, input_rank);
  TypedAllocator::Deallocate(allocator, d_output_strides, output_rank);
}

} // End of namespace tensorflow

#endif  // GOOGLE_CUDA

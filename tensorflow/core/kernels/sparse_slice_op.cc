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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_slice_op.h"

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename T>
struct SparseSliceFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_indices,
                  const Tensor& input_values, const Tensor& input_shape,
                  const Tensor& input_start, const Tensor& input_size,
                  typename AsyncOpKernel::DoneCallback done) const {
    (void)done;  // Unused (only used in GPU implementation)
    const int input_dims = input_shape.NumElements();

    sparse::SparseTensor sparse_tensor;
    TensorShape sparse_tensor_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeBase<TensorShape>::BuildTensorShapeBase(
                       input_shape.vec<int64>(), &sparse_tensor_shape));
    OP_REQUIRES_OK(context, sparse::SparseTensor::Create(
                                input_indices, input_values,
                                sparse_tensor_shape, &sparse_tensor));

    const gtl::ArraySlice<int64> start(input_start.flat<int64>().data(),
				       input_dims);
    const gtl::ArraySlice<int64> size(input_size.flat<int64>().data(),
				      input_dims);

    const StatusOr<sparse::SparseTensor> output_or =
        sparse::SparseTensor::Slice<T>(sparse_tensor, start, size);
    OP_REQUIRES_OK(context, output_or.status());
    auto output = output_or.ValueOrDie();

    context->set_output(0, output.indices());
    context->set_output(1, output.values());

    TensorShape output_shape;
    OP_REQUIRES_OK(context, TensorShapeBase<TensorShape>::BuildTensorShapeBase(
                                output.shape(), &output_shape));

    TensorShape allocated_shape;
    OP_REQUIRES_OK(context, TensorShapeBase<TensorShape>::BuildTensorShapeBase(
                                {output_shape.dims()}, &allocated_shape));

    Tensor* shape = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, allocated_shape, &shape));
    for (int dim = 0; dim < output_shape.dims(); ++dim) {
      shape->vec<int64>()(dim) = output_shape.dim_size(dim);
    }
  }
};

}  // namespace functor

namespace {

template <typename Device, typename T>
void SparseSliceOpImpl(OpKernelContext* context,
                       typename AsyncOpKernel::DoneCallback done = nullptr) {
  // Note that setting this empty lambda as the default parameter value directly
  // can cause strange compiler/linker errors, so we do it like this instead.
  if (!done) {
    done = [] {};
  }

  const Tensor& input_indices = context->input(0);
  const Tensor& input_values = context->input(1);
  const Tensor& input_shape = context->input(2);
  const Tensor& input_start = context->input(3);
  const Tensor& input_size = context->input(4);

  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
                    errors::InvalidArgument(
                        "Input indices should be a matrix but received shape ",
                        input_indices.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_values.shape()),
                    errors::InvalidArgument(
                        "Input values should be a vector but received shape ",
                        input_values.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_shape.shape()),
                    errors::InvalidArgument(
                        "Input shape should be a vector but received shape ",
                        input_shape.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_start.shape()),
                    errors::InvalidArgument(
                        "Input start should be a vector but received shape ",
                        input_start.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_size.shape()),
                    errors::InvalidArgument(
                        "Input size should be a vector but received shape ",
                        input_size.shape().DebugString()),
                    done);

  const int input_dims = input_shape.NumElements();
  OP_REQUIRES_ASYNC(context, input_dims == input_start.NumElements(),
                    errors::InvalidArgument(
                        "Expected start to be a vector of length ", input_dims,
                        " but got length ", input_start.NumElements()),
                    done);

  OP_REQUIRES_ASYNC(context, input_dims == input_size.NumElements(),
                    errors::InvalidArgument(
                        "Expected size to be a vector of length ", input_dims,
                        " but got length ", input_size.NumElements()),
                    done);

  functor::SparseSliceFunctor<Device, T>()(context, input_indices, input_values,
                                           input_shape, input_start, input_size,
                                           done);
}

}  // end of anonymous namespace

template <typename Device, typename T>
class SparseSliceOp : public OpKernel {
 public:
  explicit SparseSliceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    SparseSliceOpImpl<Device, T>(context);
  }
};

#define REGISTER_KERNELS(type)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSlice").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseSliceOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
class SparseSliceGPUOp : public AsyncOpKernel {
 public:
  explicit SparseSliceGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    SparseSliceOpImpl<GPUDevice, T>(context, done);
  }
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseSlice")             \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("shape")        \
                              .HostMemory("start")        \
                              .HostMemory("size")         \
                              .HostMemory("output_shape") \
                              .TypeConstraint<type>("T"), \
                          SparseSliceGPUOp<type>)

TF_CALL_POD_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

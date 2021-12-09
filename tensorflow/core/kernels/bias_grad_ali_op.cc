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

#include "tensorflow/core/kernels/bias_grad_ali_op_cpu.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/redux_functor.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {

void GetBiasValueDims(const Tensor& value_tensor, TensorFormat data_format,
                      int32* batch, int32* height, int32* width,
                      int32* channel) {
  *batch = 1;
  *width = 1;
  *height = 1;
  *channel = 1;
  if (data_format == FORMAT_NHWC) {
    int32 channel_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  } else if (data_format == FORMAT_NCHW) {
    int32 channel_dim = value_tensor.dims() - 3;
    int32 height_dim = value_tensor.dims() - 2;
    int32 width_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    *height = static_cast<int32>(value_tensor.dim_size(height_dim));
    *width = static_cast<int32>(value_tensor.dim_size(width_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  }
}

void GetBiasValueDims(const Tensor& value_tensor, TensorFormat data_format,
                      int32* batch, int32* height, int32* width, int32* depth,
                      int32* channel) {
  *batch = 1;
  *height = 1;
  *width = 1;
  *depth = 1;
  *channel = 1;
  if (data_format == FORMAT_NHWC) {
    int32 channel_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  } else if (data_format == FORMAT_NCHW) {
    *batch = static_cast<int32>(value_tensor.dim_size(0));
    *channel = static_cast<int32>(value_tensor.dim_size(1));
    *height = static_cast<int32>(value_tensor.dim_size(2));
    if (value_tensor.dims() > 3) {
      *width = static_cast<int32>(value_tensor.dim_size(3));
    }
    if (value_tensor.dims() > 4) {
      *depth = static_cast<int32>(value_tensor.dim_size(4));
    }
  }
}

}  // namespace

template <typename Device, typename T>
class BiasGradAliOp : public OpKernel {
 public:
  explicit BiasGradAliOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));

    OP_REQUIRES(
        context,
        FastBoundsCheck(output_backprop.NumElements(),
                        std::numeric_limits<int32>::max()),
        errors::InvalidArgument("BiasGrad requires tensor size <= int32 max"));

    int32 batch, height, width, depth, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &depth, &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (channel == 0) {
      return;  // Nothing to do
    } else if (output_backprop.NumElements() == 0) {
      // Eigen often crashes by design on empty tensors, but setZero is safe
      output->template flat<T>().setZero();
    } else {
      // Added by intel_tf to support NCHW on CPU regardless of MKL used or not.
      if (data_format_ == FORMAT_NCHW && output_backprop.dims() == 4) {
        Eigen::DSizes<Eigen::Index, 4> four_dims(batch, channel, height, width);
#ifdef EIGEN_HAS_INDEX_LIST
        using idx0 = Eigen::type2index<0>;
        using idx2 = Eigen::type2index<2>;
        using idx3 = Eigen::type2index<3>;
        Eigen::IndexList<idx0, idx2, idx3> reduction_axes;
#else
        Eigen::array<Eigen::Index, 3> reduction_axes = {0, 2, 3};
#endif
        output->template flat<T>().device(context->eigen_device<Device>()) =
            output_backprop.flat<T>()
                .template cast<typename AccumulatorType<T>::type>()
                .reshape(four_dims)
                .sum(reduction_axes)
                .template cast<T>();  // End of code by intel_tf.
      } else if (data_format_ == FORMAT_NCHW) {
        using AccumT = typename AccumulatorType<T>::type;
        const functor::ReduceMiddleDimensions<
            T, AccumT, T, Eigen::internal::scalar_sum_op<AccumT>,
            Eigen::internal::SumReducer<T>>
            redux;
        Eigen::DSizes<Eigen::Index, 3> three_dims(batch, channel,
                                                  height * width * depth);
        redux(context->eigen_device<Device>(), three_dims, output_backprop,
              output, 1);
      } else {
        Eigen::DSizes<int, 2> two_dims(batch * height * width, channel);
        functor::BiasGrad2D<Device, T> functor;
        functor(context->template eigen_device<Device>(),
                output_backprop.flat<T>(), two_dims,
                output->template flat<T>());
      }
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the CPU implementations.
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BiasGradAliOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES_NO_BFLOAT16(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow

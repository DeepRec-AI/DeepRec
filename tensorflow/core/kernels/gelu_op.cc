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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/gelu_op.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class GeluOp : public UnaryElementWiseOp<T, GeluOp<Device, T>> {
 public:
  explicit GeluOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, GeluOp<Device, T>>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("approximate", &approximate_));
  }

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Gelu<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(), approximate_,
            output->flat<T>());
  }

 private:
  bool approximate_;
};

template <typename Device, typename T>
class GeluGradOp : public BinaryElementWiseOp<T, GeluGradOp<Device, T>> {
 public:
  explicit GeluGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<T, GeluGradOp<Device, T>>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("approximate", &approximate_));
  }

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);
  // INPUTS:
  //   g (gradients): backpropagated gradients.
  //   a (inputs): inputs that were passed to GeluOp().
  // OUTPUT:
  //   gradients to backprop.
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, output);
  }

 private:
  bool approximate_;
};

template <typename Device, typename T>
void GeluGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                              const Tensor& g, const Tensor& a,
                                              Tensor* output) {
  OP_REQUIRES(context, a.IsSameSize(g),
              errors::InvalidArgument("g and a must be the same size"));
  functor::GeluGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          approximate_, output->flat<T>());
}

#define REGISTER_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Gelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      GeluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("GeluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      GeluGradOp<CPUDevice, type>);

TF_CALL_FLOAT_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                          \
  template <>                                                        \
  void Gelu<GPUDevice, T>::operator()(                               \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,  \
      bool approximate, typename TTypes<T>::Tensor activations);     \
  extern template struct Gelu<GPUDevice, T>;                         \
                                                                     \
  template <>                                                        \
  void GeluGrad<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients, \
      typename TTypes<T>::ConstTensor features, bool approximate,    \
      typename TTypes<T>::Tensor backprops);                         \
  extern template struct GeluGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Gelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      GeluOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("GeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      GeluGradOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

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

#ifdef INTEL_MKL

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"

using dnnl::primitive_attr;
using dnnl::stream;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklDequantizeOp : public OpKernel {
 public:
  explicit MklDequantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(ctx, mode_string == "SCALED",
                errors::InvalidArgument(
                    "MklDequantizeOp only supports 'SCALED' mode, but got '" +
                    mode_string + "'"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
  }

  void Compute(OpKernelContext* ctx) override {
    try {
      // Using CPU device
      auto cpu_engine = engine(ENGINE_CPU, 0);

      // Get the inputs
      const Tensor& src_tensor = ctx->input(kSrcIndex);
      const float min_range = ctx->input(1).flat<float>()(0);
      const float max_range = ctx->input(2).flat<float>()(0);

      // Get MklShape
      MklDnnShape src_mkl_shape;
      GetMklShape(ctx, kSrcIndex, &src_mkl_shape);
      auto src_tf_shape = src_mkl_shape.IsMklTensor()
                              ? src_mkl_shape.GetTfShape()
                              : src_tensor.shape();

      // src_dims is the dimension of src_tensor
      // output_dims are same as src_dims
      auto src_dims = src_mkl_shape.IsMklTensor()
                          ? src_mkl_shape.GetSizesAsMklDnnDims()
                          : TFShapeToMklDnnDims(src_tensor.shape());
      auto output_dims = src_dims;

      // Create reorder memory for src and dst
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<float> dst(&cpu_engine);

      std::shared_ptr<stream> reorder_stream;
      MklDnnThreadPool eigen_tp(ctx);
      reorder_stream.reset(CreateStream(&eigen_tp, cpu_engine));

      memory::format_tag dst_layout_type;
      switch (src_tf_shape.dims()) {
        case 1:
          dst_layout_type = memory::format_tag::x;
          break;
        case 2:
          dst_layout_type = memory::format_tag::nc;
          break;
        case 3:
          dst_layout_type = memory::format_tag::tnc;
          break;
        case 4:
          dst_layout_type = memory::format_tag::nhwc;
          break;
        case 5:
          dst_layout_type = memory::format_tag::ndhwc;
          break;
        default:
          OP_REQUIRES_OK(ctx,
                         errors::Aborted("Input dims must be <= 5 and >= 1"));
          return;
      }

      // If input is in OneDNN layout, then simply grab input layout; otherwise,
      // construct input TF layout. For TF layout, although input shape
      // (src_dims) required is in OneDNN order, the layout is Tensorflow's
      // layout
      auto src_md =
          src_mkl_shape.IsMklTensor()
              ? src_mkl_shape.GetMklLayout()
              : memory::desc(src_dims, MklDnnType<T>(), dst_layout_type);

      src.SetUsrMem(src_md, &src_tensor);
      src.SetUsrMemDataHandle(&src_tensor, reorder_stream);

      Tensor* output_tensor = nullptr;
      MklDnnShape output_mkl_shape;
      TensorShape output_tf_shape;
      memory::desc dst_md = memory::desc();
      if (src_mkl_shape.IsMklTensor()) {
        dst_md = memory::desc(src_mkl_shape.GetMklLayout().data);
        // There is no API in OneDNN v1.x to construct memory descriptor with
        // same .data field but different type.
        dst_md.data.data_type = memory::convert_to_c(MklDnnType<float>());
      } else {
        dst_md = memory::desc(src_dims, MklDnnType<float>(), dst_layout_type);
      }

      // If input is OneDNN shape, output is also OneDNN shape.
      // If input is TF shape, output is also TF shape.
      if (src_mkl_shape.IsMklTensor()) {
        output_mkl_shape.SetMklTensor(true);
        output_mkl_shape.SetMklLayout(&dst_md);
        output_mkl_shape.SetElemType(MklDnnType<float>());
        output_mkl_shape.SetTfLayout(src_mkl_shape.GetDimension(),
                                     src_mkl_shape.GetSizesAsMklDnnDims(),
                                     src_mkl_shape.GetTfDataFormat());
        output_tf_shape.AddDim(GET_MEMORY_SIZE_FROM_MD(dst_md, cpu_engine) /
                               sizeof(float));
      } else {
        output_mkl_shape.SetMklTensor(false);
        output_tf_shape = MklDnnDimsToTFShape(output_dims);
      }

      // Allocate OneDNN or TF output shape based on the above
      AllocateOutputSetMklShape(ctx, 0, &output_tensor, output_tf_shape,
                                output_mkl_shape);
      dst.SetUsrMem(dst_md, output_tensor);
      dst.SetUsrMemDataHandle(output_tensor, reorder_stream);

      // The quantization logic here for mode SCALED is similar to the logic
      // in QuantizeAndDequantizeV2 and QuantizeAndDequantizeV3.
      static constexpr int num_bits = sizeof(T) * 8;
      const float max_abs = std::max(std::abs(min_range), std::abs(max_range));
      bool is_signed = std::is_signed<T>::value;
      const int target_bits = is_signed ? (num_bits - 1) : num_bits;
      const float v_max = static_cast<float>(uint64_t{1} << target_bits) - 1;
      float v_min = 0;
      if (is_signed) {
        v_min = -(static_cast<float>(uint64_t{1} << target_bits));
      }
      if (narrow_range_) {
        v_min += 1;
      }
      float scale_factor;
      if (v_min != 0) {
        scale_factor = std::max(min_range / v_min, max_range / v_max);
      } else {
        scale_factor = max_range / v_max;
      }
      std::vector<float> scales;
      scales.push_back(scale_factor);
      primitive_attr attr;
      attr.set_output_scales(0, scales);
      std::vector<primitive> net;

      // Create reorder primitive and then execute.
      auto reorder_pd = REORDER_PD_CONSTRUCTOR_WITH_ATTR(
          GET_MEMORY_PRIMITIVE_DESC_FROM_MEM_PTR(src.GetUsrMem()),
          GET_MEMORY_PRIMITIVE_DESC_FROM_MEM_PTR(dst.GetUsrMem()), cpu_engine,
          attr);
      net.push_back(reorder(reorder_pd));
      std::vector<std::unordered_map<int, memory>> reorder_net_args;
      reorder_net_args.push_back({{DNNL_ARG_FROM, *src.GetUsrMem()},
                                  { DNNL_ARG_TO,
                                    *dst.GetUsrMem() }});
      execute_primitives(net, reorder_stream, reorder_net_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  const size_t kSrcIndex = 0;
  const size_t kMinIndex = 1;
  const size_t kMaxIndex = 2;
  bool narrow_range_;
};

REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, quint8>);
REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, qint8>);

}  // namespace tensorflow

#endif  // INTEL_MKL

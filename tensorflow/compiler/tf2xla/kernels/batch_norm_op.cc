/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// XLA implementation of BatchNorm operations.
#include "tensorflow/compiler/tf2xla/kernels/relu_op.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class FusedBatchNormOp : public XlaOpKernel {
 public:
  explicit FusedBatchNormOp(OpKernelConstruction* ctx)
      : FusedBatchNormOp(ctx, false) {}

  FusedBatchNormOp(OpKernelConstruction* ctx, bool is_batch_norm_ex)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(
        ctx, FormatFromString(data_format_str, &data_format_),
        errors::InvalidArgument("Invalid data format: ", data_format_str));

    if (is_batch_norm_ex) {
      int num_side_inputs;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_side_inputs", &num_side_inputs));
      OP_REQUIRES(ctx, num_side_inputs >= 0 && num_side_inputs <= 1,
                  errors::InvalidArgument(
                      "FusedBatchNormEx supports at most 1 side input."));
      add_side_input_ = (num_side_inputs == 1);
      string activation_mode;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_mode", &activation_mode));
      OP_REQUIRES(ctx,
                  activation_mode == "Identity" || activation_mode == "Relu",
                  errors::InvalidArgument(
                      "Unsupported FusedBatchNormEx activation mode: ",
                      activation_mode));
      apply_relu_ = (activation_mode == "Relu");
    } else {
      add_side_input_ = false;
      apply_relu_ = false;
    }
    is_on_gpu_ = ctx->device_type().type_string() == DEVICE_GPU_XLA_JIT;
  }

  void Compile(XlaOpKernelContext* ctx) override { CompileImpl(ctx, false); }

 protected:
  virtual void CompileImpl(XlaOpKernelContext* ctx, bool use_reserved_space) {
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));
    xla::PrimitiveType scale_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(1), &scale_type));

    xla::XlaOp input = ctx->Input(0);
    TensorShape input_shape = ctx->InputShape(0);

    int feature_index =
        GetTensorFeatureDimIndex(input_shape.dims(), data_format_);

    // TODO(b/69928690): support mixed precision in the XLA batch normalization
    // operators. As a workaround, cast everything to the statistics type (which
    // may be more precise than the input type).
    input = xla::ConvertElementType(input, scale_type);
    if (is_training_) {
      size_t reserve_space_size = 0;
      if (is_on_gpu_ && use_reserved_space) {
        OpKernelContext* opkernel_ctx = ctx->op_kernel_context();
        // The device is an XlaCompilation device which is a 'dummy' TensorFlow
        // device that is only used to execute a
        // subgraph of XLA compilation Ops to construct a compiled version
        // of the subgraph's computation.
        CHECK_NE(opkernel_ctx->device(), nullptr);
        VLOG(2) << "XlaCompilation Device* " << opkernel_ctx->device();
        // Stream (shared by the XlaDevice) is set during the construction of
        // the above device that can be used here.
        se::Stream* stream =
            opkernel_ctx->device()->get_gpu_device_info_stream();
        VLOG(2) << "Stream* "
                << opkernel_ctx->device()->get_gpu_device_info_stream();
        int64 batch_index =
            GetTensorBatchDimIndex(input_shape.dims(), data_format_);
        int64 batch_size = input_shape.dim_size(batch_index);
        int64 feature_count = input_shape.dim_size(feature_index);

        int num_spatial_dims =
            GetTensorSpatialDims(input_shape.dims(), data_format_);
        int64 y_size = 1;
        for (int i = 0; i < num_spatial_dims; i++) {
          y_size *= input_shape.dim_size(
              GetTensorSpatialDimIndex(input_shape.dims(), data_format_, i));
        }
        std::string input_type_str =
            (input_type == xla::PrimitiveType::F16) ? "F16" : "F32";
        std::string scale_type_str =
            (scale_type == xla::PrimitiveType::F32) ? "F32" : "F16";
        // std::string input_type_str_1 = ToString(input_type);
        VLOG(1) << "Batch index: " << batch_index
                << " Batch Size: " << batch_size
                << " Feature index: " << feature_index
                << " Feature count: " << feature_count
                << " Num Dims: " << input_shape.dims()
                << " Input type: " << input_type_str
                << " Scale type: " << scale_type_str << " y_size = " << y_size;

        if (input_type == xla::PrimitiveType::F16) {
          stream->ThenFindBatchNormalizationTrainingExReserveSpaceSize<
              Eigen::half>(batch_size, feature_count, y_size,
                           ::tensorflow::ToString(data_format_),
                           &reserve_space_size);
        } else if (input_type == xla::PrimitiveType::F32) {
          stream->ThenFindBatchNormalizationTrainingExReserveSpaceSize<float>(
              batch_size, feature_count, y_size,
              ::tensorflow::ToString(data_format_), &reserve_space_size);
        } else {
          errors::Unimplemented("Unimplemented data type for batchnorm input");
        }
        VLOG(1) << "Reserved space required: " << reserve_space_size
                << " bytes";
      }
      xla::XlaOp output = xla::BatchNormTraining(
          input, ctx->Input(1), ctx->Input(2), epsilon_, feature_index,
          reserve_space_size, use_reserved_space);

      // In training mode, outputs the normalized value as well as the
      // calculated mean and variance. Optionally we add side input and apply
      // relu activation.
      xla::XlaOp converted =
          xla::ConvertElementType(xla::GetTupleElement(output, 0), input_type);
      if (add_side_input_ && apply_relu_) {
        ctx->SetOutput(0, xla::Relu(xla::Add(ctx->Input(5), converted)));
      } else if (apply_relu_) {
        ctx->SetOutput(0, xla::Relu(converted));
      } else {
        ctx->SetOutput(0, converted);
      }

      ctx->SetOutput(1, xla::GetTupleElement(output, 1));
      xla::XlaOp variance = xla::GetTupleElement(output, 2);
      // Apply Bessel's correction.
      int total_input_size = ctx->InputShape(0).num_elements();
      int total_scale_size = ctx->InputShape(1).num_elements();
      int sample_size = total_input_size / total_scale_size;
      int sample_size_minus_one = std::max(1, sample_size - 1);
      double factor = static_cast<double>(sample_size) /
                      static_cast<double>(sample_size_minus_one);

      xla::XlaOp corrected =
          xla::Mul(variance, xla::ScalarLike(variance, factor));
      ctx->SetOutput(2, corrected);

      // Output 3 and 4 for "FusedBatchNorm" are currently marked as "reserved
      // space 1 & 2". They are used to pass the per-batch mean and
      // variance to the gradient. Here we maintain the same behavior by setting
      // them to the mean and variance calculated by BatchNormTraining.
      ctx->SetOutput(3, xla::GetTupleElement(output, 1));
      if (is_on_gpu_) {
        // The last two outputs from the FusedBatchNorm training TensorFlow GPU
        // op are implementation defined.  For now we rely on the in-practice
        // behavior of the op:
        //   output 3 is the mean
        //   output 4 is rsqrt(variance + epsilon)
        ctx->SetOutput(4, xla::Rsqrt(xla::Add(
                              variance, xla::ScalarLike(variance, epsilon_))));
      } else {
        ctx->SetOutput(4, variance);
      }
      batch_norm_training_ = output;
    } else {
      xla::XlaOp output = xla::BatchNormInference(
          input, ctx->Input(1), ctx->Input(2), ctx->Input(3), ctx->Input(4),
          epsilon_, feature_index);

      xla::XlaOp converted = xla::ConvertElementType(output, input_type);
      if (add_side_input_ && apply_relu_) {
        ctx->SetOutput(0, xla::Relu(xla::Add(ctx->Input(5), converted)));
      } else if (apply_relu_) {
        ctx->SetOutput(0, xla::Relu(converted));
      } else {
        ctx->SetOutput(0, converted);
      }

      // Directly send input to output as mean and variance in inference mode.
      ctx->SetOutput(1, ctx->Input(3));
      ctx->SetOutput(2, ctx->Input(4));
      ctx->SetOutput(3, ctx->Input(3));
      ctx->SetOutput(4, ctx->Input(4));
    }
  }
protected:
  xla::XlaOp batch_norm_training_;

 private:
  float epsilon_;
  TensorFormat data_format_;
  bool is_training_;
  bool add_side_input_;
  bool apply_relu_;
  bool is_on_gpu_;
};

class FusedBatchNormOpV3 : public FusedBatchNormOp {
 public:
  explicit FusedBatchNormOpV3(OpKernelConstruction* ctx)
      : FusedBatchNormOp(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    FusedBatchNormOp::CompileImpl(ctx, true);
    if (!ctx->status().ok()) {
      return;
    }
    ctx->SetOutput(5, xla::GetTupleElement(batch_norm_training_, 3));
  }
};

class FusedBatchNormOpEx : public FusedBatchNormOp {
 public:
  explicit FusedBatchNormOpEx(OpKernelConstruction* ctx)
      : FusedBatchNormOp(ctx, /*is_batch_norm_ex=*/true) {}

  void Compile(XlaOpKernelContext* ctx) override {
    FusedBatchNormOp::CompileImpl(ctx, true);
    if (!ctx->status().ok()) {
      return;
    }
    ctx->SetOutput(5, xla::GetTupleElement(batch_norm_training_, 3));
  }
};

REGISTER_XLA_OP(Name("FusedBatchNorm"), FusedBatchNormOp);
REGISTER_XLA_OP(Name("FusedBatchNormV2"), FusedBatchNormOp);
REGISTER_XLA_OP(Name("FusedBatchNormV3"), FusedBatchNormOpV3);
REGISTER_XLA_OP(Name("_FusedBatchNormEx"), FusedBatchNormOpEx);

class FusedBatchNormGradOp : public XlaOpKernel {
 public:
  explicit FusedBatchNormGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(
        ctx, FormatFromString(data_format_str, &data_format_),
        errors::InvalidArgument("Invalid data format: ", data_format_str));
    is_on_gpu_ = ctx->device_type().type_string() == DEVICE_GPU_XLA_JIT;
  }
  void Compile(XlaOpKernelContext* ctx) override { CompileImpl(ctx, false); }

  virtual void CompileImpl(XlaOpKernelContext* ctx, bool use_reserved_space ) {
    xla::XlaBuilder* const b = ctx->builder();
    DataType input_dtype = ctx->input_type(0);
    DataType scale_dtype = ctx->input_type(2);

    // TODO(b/69928690): support mixed precision in the XLA batch normalization
    // operators. For now, cast everything to the statistics type (which
    // may be more precise than the input type).
    auto grad_backprop =
        XlaHelpers::ConvertElementType(ctx->Input(0), scale_dtype);
    auto activations =
        XlaHelpers::ConvertElementType(ctx->Input(1), scale_dtype);
    auto scale = ctx->Input(2);
    auto mean = ctx->Input(3);
    auto var = ctx->Input(4);

    const int input_dims = ctx->InputShape(0).dims();
    const int feature_index =
        GetTensorFeatureDimIndex(input_dims, data_format_);

    xla::XlaOp x_backprop;
    xla::XlaOp scale_backprop;
    xla::XlaOp offset_backprop;
    if (is_training_) {
      if (is_on_gpu_) {
        // The last two inputs to the FusedBatchNormGrad training TensorFlow GPU
        // op are implementation defined.  For now we rely on the in-practice
        // behavior of the op: input 3 is the mean input 4 is rsqrt(variance +
        // epsilon)
        //
        // The XLA op expects:
        //   input 3 is the mean
        //   input 4 is the variance
        //
        // so we adjust input 4 here.
        xla::XlaOp one = xla::ScalarLike(var, 1.0f);
        xla::XlaOp epsilon = xla::ScalarLike(var, epsilon_);
        var = xla::Sub(one / (var * var), epsilon);
      }
      xla::XlaOp reserve_space;
      if (use_reserved_space) {
        reserve_space = ctx->Input(5);
      }
      xla::XlaOp output =
          xla::BatchNormGrad(activations, scale, mean, var, grad_backprop, reserve_space, 
                             epsilon_, feature_index);
      x_backprop = xla::GetTupleElement(output, 0);
      scale_backprop = xla::GetTupleElement(output, 1);
      offset_backprop = xla::GetTupleElement(output, 2);
    } else {
      // Reduce over all dimensions except the feature dim.
      std::vector<int64> reduction_dims(input_dims - 1);
      std::iota(reduction_dims.begin(), reduction_dims.begin() + feature_index,
                0);
      std::iota(reduction_dims.begin() + feature_index, reduction_dims.end(),
                feature_index + 1);
      // offset_backprop  = sum(y_backprop)
      // scale_backprop = y_backprop * ((x - pop_mean) * rsqrt(pop_var +
      // epsilon))
      // x_backprop = y_backprop * (scale * rsqrt(pop_var + epsilon))
      const DataType accumulation_type =
          XlaHelpers::SumAccumulationType(scale_dtype);
      auto converted =
          XlaHelpers::ConvertElementType(grad_backprop, accumulation_type);
      auto reduce =
          xla::Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                      *ctx->GetOrCreateAdd(accumulation_type), reduction_dims);
      offset_backprop = XlaHelpers::ConvertElementType(reduce, scale_dtype);

      // scratch1 = rsqrt(pop_var + epsilon)
      auto epsilon = XlaHelpers::FloatLiteral(b, scale_dtype, epsilon_);
      auto scratch1 = xla::Rsqrt(xla::Add(var, epsilon));

      // scratch2 = sum(y_backprop * (x - mean))
      auto mul =
          xla::Mul(grad_backprop, xla::Sub(activations, mean, {feature_index}));
      converted = XlaHelpers::ConvertElementType(mul, accumulation_type);
      reduce =
          xla::Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                      *ctx->GetOrCreateAdd(accumulation_type), reduction_dims);
      auto scratch2 = XlaHelpers::ConvertElementType(reduce, scale_dtype);

      x_backprop =
          xla::Mul(grad_backprop, xla::Mul(scratch1, scale), {feature_index});
      scale_backprop = xla::Mul(scratch1, scratch2);
    }

    ctx->SetOutput(0, XlaHelpers::ConvertElementType(x_backprop, input_dtype));
    ctx->SetOutput(1, scale_backprop);
    ctx->SetOutput(2, offset_backprop);
    ctx->SetConstantOutput(3, Tensor());
    ctx->SetConstantOutput(4, Tensor());
  }

 private:
  TensorFormat data_format_;
  float epsilon_;
  bool is_training_;
  bool is_on_gpu_;
};

class FusedBatchNormGradOpV3 : public FusedBatchNormGradOp {
 public:
  explicit FusedBatchNormGradOpV3(OpKernelConstruction* ctx)
      : FusedBatchNormGradOp(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    FusedBatchNormGradOp::CompileImpl(ctx, true);
    if (!ctx->status().ok()) {
      return;
    }
  }
};

REGISTER_XLA_OP(Name("FusedBatchNormGrad"), FusedBatchNormGradOp);
REGISTER_XLA_OP(Name("FusedBatchNormGradV2"), FusedBatchNormGradOp);
REGISTER_XLA_OP(Name("FusedBatchNormGradV3"), FusedBatchNormGradOpV3);

}  // namespace
}  // namespace tensorflow

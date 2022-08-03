/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/ctc_ops.cc.

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif // GOOGLE_CUDA

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/ctc/ctc_loss_calculator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/stream_executor_util.h"
#endif // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA
using GPUDevice = Eigen::GpuDevice;

namespace {
using se::DeviceMemory;
using se::Stream;
using se::StreamExecutor;
using se::ScratchAllocator;
using se::dnn::CtcLossDescriptor;
using se::dnn::RnnStateTensorDescriptor;
using se::dnn::ToDataType;
using se::port::StatusOr;

template<typename T>
void DoHistogram(OpKernelContext* ctx, const Tensor* labels_indices,
                 int num_indices, int batch_size,
                 std::vector<int> *labels_lengths) {
  const T* h_in = labels_indices->flat<T>().data();
  for(int i = 0; i < num_indices; i++) {
    T key = h_in[i * 2];
    (*labels_lengths)[key]++;
  }
}

// A helper to allocate temporary scratch memory for cudnnCTCLoss ops. It
// takes the ownership of the underlying memory. The expectation is that the
// memory should be alive for the span of the cudnnCTCLoss itself.
template <typename T>
class CudnnCtcLossAllocatorInTemp : public ScratchAllocator {
 public:
  ~CudnnCtcLossAllocatorInTemp() override = default;

  explicit CudnnCtcLossAllocatorInTemp(OpKernelContext* context)
      : context_(context) {}

  int64 GetMemoryLimitInBytes() override {
    return std::numeric_limits<int64>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64 byte_size) override {
    Tensor temporary_memory;
    const DataType tf_data_type = DataTypeToEnum<T>::v();
    int64 allocate_count =
        Eigen::divup(byte_size, static_cast<int64>(sizeof(T)));
    Status allocation_status(context_->allocate_temp(
        tf_data_type, TensorShape({allocate_count}), &temporary_memory));
    if (!allocation_status.ok()) {
      return allocation_status;
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return DeviceMemory<uint8>::MakeFromByteSize(
        temporary_memory.template flat<T>().data(),
        temporary_memory.template flat<T>().size() * sizeof(T));
  }

  int64 TotalByteSize() const { return total_byte_size_; }

  Tensor get_allocated_tensor(int index) const {
    return allocated_tensors_[index];
  }

 private:
  int64 total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  std::vector<Tensor> allocated_tensors_;
};
} // end namespace
#endif // GOOGLE_CUDA

template <typename T>
class CTCLossOp : public OpKernel {
  typedef Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
      InputMap;
  typedef Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
      OutputMap;

 public:
  explicit CTCLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated",
                                     &preprocess_collapse_repeated_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("ctc_merge_repeated", &ctc_merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ignore_longer_outputs_than_inputs",
                                     &ignore_longer_outputs_than_inputs_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs;
    const Tensor* labels_indices;
    const Tensor* labels_values;
    const Tensor* seq_len;
    OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs));
    OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
    OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
    OP_REQUIRES_OK(ctx, ctx->input("sequence_length", &seq_len));

    OP_REQUIRES(ctx, inputs->shape().dims() == 3,
                errors::InvalidArgument("inputs is not a 3-Tensor"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(seq_len->shape()),
                errors::InvalidArgument("sequence_length is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(labels_indices->shape()),
                errors::InvalidArgument("labels_indices is not a matrix"));
    OP_REQUIRES(ctx, labels_indices->dim_size(1) > 1,
                errors::InvalidArgument(
                    "labels_indices second dimension must be >= 1. Received ",
                    labels_indices->dim_size(1)));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels_values->shape()),
                errors::InvalidArgument("labels_values is not a vector"));

    const TensorShape& inputs_shape = inputs->shape();
    const int64 max_time = inputs_shape.dim_size(0);
    OP_REQUIRES(ctx, max_time != 0,
                errors::InvalidArgument(
                    "Max time or first dimension of input cannot be 0."));
    const int64 batch_size = inputs_shape.dim_size(1);
    const int64 num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);

    OP_REQUIRES(
        ctx, batch_size == seq_len->dim_size(0),
        errors::InvalidArgument("len(sequence_length) != batch_size.  ",
                                "len(sequence_length):  ", seq_len->dim_size(0),
                                " batch_size: ", batch_size));
    auto seq_len_t = seq_len->vec<int32>();

    OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                errors::InvalidArgument(
                    "labels_indices and labels_values must contain the "
                    "same number of rows, but saw shapes: ",
                    labels_indices->shape().DebugString(), " vs. ",
                    labels_values->shape().DebugString()));

    OP_REQUIRES(ctx, batch_size != 0,
                errors::InvalidArgument("batch_size must not be 0"));

    // Figure out the maximum label length to use as sparse tensor dimension.
    auto labels_indices_t = labels_indices->matrix<int64>();
    int64 max_label_len = 0;
    for (int i = 0; i < labels_indices->dim_size(0); i++) {
      max_label_len = std::max(max_label_len, labels_indices_t(i, 1) + 1);
    }

    TensorShape labels_shape({batch_size, max_label_len});
    std::vector<int64> order{0, 1};
    sparse::SparseTensor labels_sp;
    OP_REQUIRES_OK(
        ctx, sparse::SparseTensor::Create(*labels_indices, *labels_values,
                                          labels_shape, order, &labels_sp));

    Status labels_sp_valid = labels_sp.IndicesValid();
    OP_REQUIRES(ctx, labels_sp_valid.ok(),
                errors::InvalidArgument("label SparseTensor is not valid: ",
                                        labels_sp_valid.error_message()));

    typename ctc::CTCLossCalculator<T>::LabelSequences labels_t(batch_size);
    for (const auto& g : labels_sp.group({0})) {  // iterate by batch
      const int64 batch_indices = g.group()[0];
      OP_REQUIRES(ctx, FastBoundsCheck(batch_indices, batch_size),
                  errors::InvalidArgument("labels batch index must be between ",
                                          0, " and ", batch_size,
                                          " but saw: ", batch_indices));

      auto values = g.values<int32>();
      std::vector<int>* b_values = &labels_t[batch_indices];
      b_values->resize(values.size());
      for (int i = 0; i < values.size(); ++i) (*b_values)[i] = values(i);
    }

    OP_REQUIRES(ctx, static_cast<size_t>(batch_size) == labels_t.size(),
                errors::InvalidArgument("len(labels) != batch_size.  ",
                                        "len(labels):  ", labels_t.size(),
                                        " batch_size: ", batch_size));

    for (int64 b = 0; b < batch_size; ++b) {
      OP_REQUIRES(
          ctx, seq_len_t(b) <= max_time,
          errors::InvalidArgument("sequence_length(", b, ") <= ", max_time));
    }

    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len->shape(), &loss));
    auto loss_t = loss->vec<T>();

    Tensor* gradient;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("gradient", inputs_shape, &gradient));
    auto gradient_t = gradient->tensor<T, 3>();
    auto inputs_t = inputs->tensor<T, 3>();
    std::vector<OutputMap> gradient_list_t;
    std::vector<InputMap> input_list_t;

    for (std::size_t t = 0; t < max_time; ++t) {
      input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                batch_size, num_classes);
      gradient_list_t.emplace_back(
          gradient_t.data() + t * batch_size * num_classes, batch_size,
          num_classes);
    }

    gradient_t.setZero();

    // Assumption: the blank index is num_classes - 1
    ctc::CTCLossCalculator<T> ctc_loss_calculator(num_classes - 1, 0);
    DeviceBase::CpuWorkerThreads workers =
        *ctx->device()->tensorflow_cpu_worker_threads();
    OP_REQUIRES_OK(ctx, ctc_loss_calculator.CalculateLoss(
                            seq_len_t, labels_t, input_list_t,
                            preprocess_collapse_repeated_, ctc_merge_repeated_,
                            ignore_longer_outputs_than_inputs_, &loss_t,
                            &gradient_list_t, &workers));
  }

 private:
  bool preprocess_collapse_repeated_;
  bool ctc_merge_repeated_;
  bool ignore_longer_outputs_than_inputs_;

  TF_DISALLOW_COPY_AND_ASSIGN(CTCLossOp<T>);
};

#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CTCLoss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CTCLossOp<T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

#if GOOGLE_CUDA
class CTCLossOpGPU : public OpKernel {

 public:
  explicit CTCLossOpGPU(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated",
                                     &preprocess_collapse_repeated_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("ctc_merge_repeated", &ctc_merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ignore_longer_outputs_than_inputs",
                                     &ignore_longer_outputs_than_inputs_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs;
    const Tensor* labels_indices;
    const Tensor* labels_values;
    const Tensor* seq_len;
    OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs));
    OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
    OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
    OP_REQUIRES_OK(ctx, ctx->input("sequence_length", &seq_len));

    OP_REQUIRES(ctx, inputs->shape().dims() == 3,
                errors::InvalidArgument("inputs is not a 3-Tensor"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(seq_len->shape()),
                errors::InvalidArgument("sequence_length is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(labels_indices->shape()),
                errors::InvalidArgument("labels_indices is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels_values->shape()),
                errors::InvalidArgument("labels_values is not a vector"));

    const TensorShape& inputs_shape = inputs->shape();
    const int64 max_time_raw = inputs_shape.dim_size(0);
    const int64 batch_size_raw = inputs_shape.dim_size(1);
    const int64 num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int max_time = static_cast<const int>(max_time_raw);
    const int batch_size = static_cast<const int>(batch_size_raw);
    const int num_classes = static_cast<const int>(num_classes_raw);

    OP_REQUIRES(
        ctx, batch_size == seq_len->dim_size(0),
        errors::InvalidArgument("len(sequence_length) != batch_size.  ",
                                "len(sequence_length):  ", seq_len->dim_size(0),
                                " batch_size: ", batch_size));

    OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                errors::InvalidArgument(
                    "labels_indices and labels_values must contain the "
                    "same number of rows, but saw shapes: ",
                    labels_indices->shape().DebugString(), " vs. ",
                    labels_values->shape().DebugString()));
    auto num_indices = labels_indices->dim_size(0);

    OP_REQUIRES(ctx, batch_size != 0,
                errors::InvalidArgument("batch_size must not be 0"));

    
    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len->shape(), &loss));

    Tensor* gradient = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("gradient", inputs_shape, &gradient));

    OP_REQUIRES(ctx, preprocess_collapse_repeated_ == false,
                errors::InvalidArgument("GPU CTCLossOp requires "
                                        "preprocess_collapse_repeated to be "
                                        "false"));
    OP_REQUIRES(ctx, ctc_merge_repeated_ == true,
                errors::InvalidArgument("GPU CTCLossOp requires "
                                        "ctc_merge_repeated_ to be "
                                        "true"));
    OP_REQUIRES(ctx, ignore_longer_outputs_than_inputs_ == false,
                errors::InvalidArgument("GPU CTCLossOp requires "
                                        "ignore_longer_outputs_than_inputs_ to"
                                        "be false"));

    // Convert the labels_indices to labels_lengths
    std::vector<int> labels_lengths(batch_size, 0);
    DoHistogram<int64>(ctx, labels_indices, num_indices, batch_size,
                       &labels_lengths);

    StreamExecutor* executor = ctx->op_device_context()->stream()->parent();
    se::dnn::DataType data_type = ToDataType<float>::value;

    std::unique_ptr<CtcLossDescriptor> ctc_loss_desc;
    std::unique_ptr<RnnStateTensorDescriptor> probs_desc;
    std::unique_ptr<RnnStateTensorDescriptor> grads_desc;

    auto ctc_loss_desc_s = executor->createCtcLossDescriptor(data_type);
    OP_REQUIRES_OK(ctx, ctc_loss_desc_s.status());
    ctc_loss_desc = ctc_loss_desc_s.ConsumeValueOrDie();
    
    auto probs_desc_s = executor->createRnnStateTensorDescriptor(
        max_time, batch_size, num_classes, data_type);
    OP_REQUIRES_OK(ctx, probs_desc_s.status());
    probs_desc = probs_desc_s.ConsumeValueOrDie();

    auto grads_desc_s = executor->createRnnStateTensorDescriptor(
        max_time, batch_size, num_classes, data_type);
    OP_REQUIRES_OK(ctx, grads_desc_s.status());
    grads_desc = grads_desc_s.ConsumeValueOrDie();

    absl::Span<const int32> labels_data;
    absl::Span<const int32> labels_lengths_data;
    absl::Span<const int32> input_lengths_data;
    labels_data = absl::Span<const int32>(
        labels_values->flat<int32>().data(), num_indices);
    labels_lengths_data = absl::Span<const int32>(
        labels_lengths.data(), batch_size);
    input_lengths_data = absl::Span<const int32>(
        seq_len->flat<int32>().data(), batch_size);

    auto probs_data = StreamExecutorUtil::AsDeviceMemory<float>(*inputs);
    auto costs_data = StreamExecutorUtil::AsDeviceMemory<float>(*loss);
    auto grads_data = StreamExecutorUtil::AsDeviceMemory<float>(*gradient);

    CudnnCtcLossAllocatorInTemp<uint8> workspace_allocator(ctx);

    Stream* stream = ctx->op_device_context()->stream();
    bool cudnn_launch_status =
        stream
            ->ThenCtcLoss(
                *probs_desc, probs_data, labels_data, labels_lengths_data,
                input_lengths_data, &costs_data, *grads_desc, &grads_data,
                *ctc_loss_desc, &workspace_allocator)
            .ok();

    if (!cudnn_launch_status) {
      ctx->SetStatus(
          errors::Internal("cuDNN CTCLoss launch failure"));
    }
  }

 private:
  bool preprocess_collapse_repeated_;
  bool ctc_merge_repeated_;
  bool ignore_longer_outputs_than_inputs_;

  TF_DISALLOW_COPY_AND_ASSIGN(CTCLossOpGPU);
};

REGISTER_KERNEL_BUILDER(Name("CTCLossV2").Device(DEVICE_GPU)
                                       .HostMemory("labels_indices")
                                       .HostMemory("labels_values")
                                       .HostMemory("sequence_length"),
                        CTCLossOpGPU);
#endif // GOOGLE_CUDA
}  // end namespace tensorflow

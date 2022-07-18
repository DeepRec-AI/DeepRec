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

// See docs in ../ops/array_ops.cc.

#include <limits>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

#ifdef INTEL_MKL
#include "dnnl.hpp"
#endif  // INTEL_MKL

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

// --------------------------------------------------------------------------
template <typename Device, typename T, AxisArgumentName AxisArgName>
class ConcatBaseOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit ConcatBaseOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    const Tensor* concat_dim_tensor;
    const char* axis_attribute_name =
        AxisArgName == NAME_IS_AXIS ? "axis" : AxisArgName == NAME_IS_CONCAT_DIM
                                                   ? "concat_dim"
                                                   : "<invalid>";
    OP_REQUIRES_OK(c, c->input(axis_attribute_name, &concat_dim_tensor));
    OP_REQUIRES(c, IsLegacyScalar(concat_dim_tensor->shape()),
                errors::InvalidArgument(
                    axis_attribute_name,
                    " tensor should be a scalar integer, but got shape ",
                    concat_dim_tensor->shape().DebugString()));
    int64 concat_dim;
    // In case of ConcatV2, "axis" could be int32 or int64
    if (AxisArgName == NAME_IS_AXIS) {
      OP_REQUIRES(
          c,
          (concat_dim_tensor->dtype() == DT_INT32 ||
           concat_dim_tensor->dtype() == DT_INT64),
          errors::InvalidArgument(axis_attribute_name,
                                  " tensor should be int32 or int64, but got ",
                                  DataTypeString(concat_dim_tensor->dtype())));
    } else {
      OP_REQUIRES(c, (concat_dim_tensor->dtype() == DT_INT32),
                  errors::InvalidArgument(
                      axis_attribute_name, " tensor should be int32, but got ",
                      DataTypeString(concat_dim_tensor->dtype())));
    }
    if (concat_dim_tensor->dtype() == DT_INT32) {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor->scalar<int32>()());
    } else {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor->scalar<int64>()());
    }

    OpInputList values;
    OP_REQUIRES_OK(c, c->input_list("values", &values));
    const int N = values.size();
    const int input_dims = values[0].dims();
    const TensorShape& input_shape = values[0].shape();

    int32 axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(c,
                (0 <= axis && axis < input_dims) ||
                    (allow_legacy_scalars() && concat_dim == 0),
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));
    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
    // the dimension indicated with size y0, we flatten it to {x, y}, where y =
    // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(N);
    int64 inputs_flat_dim0 = 1;
    for (int d = 0; d < axis; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    int64 output_concat_dim = 0;
    const bool input_is_scalar = IsLegacyScalar(input_shape);
    for (int i = 0; i < N; ++i) {
      const auto& in = values[i];
      const bool in_is_scalar = IsLegacyScalar(in.shape());
      OP_REQUIRES(
          c, in.dims() == input_dims || (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", in.shape().DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == axis) {
          continue;
        }
        OP_REQUIRES(
            c, in.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                input_shape.DebugString(), " vs. shape[", i,
                "] = ", in.shape().DebugString()));
      }
      if (in.NumElements() > 0) {
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            in.shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
      }
      // TODO(irving): Remove check once !allow_legacy_scalars().
      output_concat_dim += in.dims() > 0 ? in.dim_size(axis) : 1;
    }

    TensorShape output_shape(input_shape);
    // TODO(irving): Remove rank 0 case once !allow_legacy_scalars().
    if (output_shape.dims() == 0) {
      output_shape.AddDim(output_concat_dim);
    } else {
      output_shape.set_dim(axis, output_concat_dim);
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() > 0) {
      int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
      auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      if (std::is_same<Device, GPUDevice>::value) {
        ConcatGPU<T>(c, inputs_flat, output, &output_flat);
        return;
      }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#ifdef TENSORFLOW_USE_SYCL
      if (std::is_same<Device, SYCLDevice>::value) {
        ConcatSYCL<T>(c->eigen_sycl_device(), inputs_flat, &output_flat);
        return;
      }
#endif  // TENSORFLOW_USE_SYCL
      ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
    }
  }
};

template <typename Device, typename T>
using ConcatOp = ConcatBaseOp<Device, T, NAME_IS_CONCAT_DIM>;
template <typename Device, typename T>
using ConcatV2Op = ConcatBaseOp<Device, T, NAME_IS_AXIS>;

#define REGISTER_CONCAT(type)                            \
  REGISTER_KERNEL_BUILDER(Name("Concat")                 \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          ConcatOp<CPUDevice, type>)     \
  REGISTER_KERNEL_BUILDER(Name("ConcatV2")               \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("axis"),       \
                          ConcatV2Op<CPUDevice, type>)

TF_CALL_POD_STRING_TYPES(REGISTER_CONCAT);
REGISTER_CONCAT(quint8);
REGISTER_CONCAT(qint8);
REGISTER_CONCAT(quint16);
REGISTER_CONCAT(qint16);
REGISTER_CONCAT(qint32);
REGISTER_CONCAT(uint32);
REGISTER_CONCAT(uint64);

#undef REGISTER_CONCAT

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Concat")                 \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          ConcatOp<GPUDevice, type>)     \
  REGISTER_KERNEL_BUILDER(Name("ConcatV2")               \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("axis"),       \
                          ConcatV2Op<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
TF_CALL_uint8(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
REGISTER_GPU(bool);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Concat")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("concat_dim")
                            .HostMemory("values")
                            .HostMemory("output"),
                        ConcatOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("values")
                            .HostMemory("axis")
                            .HostMemory("output"),
                        ConcatV2Op<CPUDevice, int32>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Concat")                 \
                              .Device(DEVICE_SYCL)       \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          ConcatOp<SYCLDevice, type>)    \
  REGISTER_KERNEL_BUILDER(Name("ConcatV2")               \
                              .Device(DEVICE_SYCL)       \
                              .TypeConstraint<type>("T") \
                              .HostMemory("axis"),       \
                          ConcatV2Op<SYCLDevice, type>)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL);

REGISTER_KERNEL_BUILDER(Name("Concat")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("concat_dim")
                            .HostMemory("values")
                            .HostMemory("output"),
                        ConcatOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("values")
                            .HostMemory("axis")
                            .HostMemory("output"),
                        ConcatV2Op<CPUDevice, int32>);

#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL

class ConcatOffsetOp : public OpKernel {
 public:
  explicit ConcatOffsetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& concat_dim = ctx->input(0);
    OP_REQUIRES(
        ctx, IsLegacyScalar(concat_dim.shape()),
        errors::InvalidArgument(
            "Concat dim tensor should be a scalar integer, but got shape ",
            concat_dim.shape().DebugString()));
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      const Tensor& inp = ctx->input(i);
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(inp.shape()),
                  errors::InvalidArgument("input ", i,
                                          " should be a vector, but got shape ",
                                          inp.shape().DebugString()));
    }
    // Suppose a Concat() op needs to Concatenate N tensors, each of
    // which has the same number of dimensions.  Their shapes match
    // except the concat dimension.
    //
    // E.g., say, we want to concatenate 3 tensors in the 2nd
    // dimension, and their shapes are:
    //
    //  [2, 2, 5, 7]
    //  [2, 3, 5, 7]
    //  [2, 4, 5, 7]
    //
    // Here, N=3, cdim=1, dims=4. The concatenated tensor has shape
    // [2,9,5,7]. We will compute the cumulative sum along the 2nd
    // dimension to figure out each input's offset in the concatenated
    // output:
    //  [0, 0, 0, 0]
    //  [0, 2, 0, 0]
    //  [0, 5, 0, 0]
    const int32 N = ctx->num_inputs() - 1;
    const Tensor& inp0 = ctx->input(1);
    auto inp0_vec = inp0.vec<int32>();
    const int64 cdim = internal::SubtleMustCopy(concat_dim.scalar<int32>()());
    const int64 dims = inp0.NumElements();
    int32 axis = cdim < 0 ? cdim + dims : cdim;
    OP_REQUIRES(ctx, FastBoundsCheck(axis, dims),
                errors::InvalidArgument("Concat dim is out of range: ", cdim,
                                        " vs. ", dims));
    int32 offset = 0;
    for (int i = 0; i < N; ++i) {
      const Tensor& inp = ctx->input(1 + i);
      OP_REQUIRES(
          ctx, dims == inp.NumElements(),
          errors::InvalidArgument("input ", i, " should contain ", dims,
                                  " elements, but got ", inp.NumElements()));
      auto inp_vec = inp.vec<int32>();
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {dims}, &out));
      auto out_vec = out->vec<int32>();
      for (int64 j = 0; j < dims; ++j) {
        if (j == axis) {
          out_vec(j) = offset;
          offset += inp_vec(j);
        } else {
          OP_REQUIRES(ctx, (inp0_vec(j) == inp_vec(j)),
                      errors::InvalidArgument(
                          "All dimensions except ", axis, " must match. Input ",
                          i, " has shape [", inp.SummarizeValue(10),
                          "] and doesn't match input 0 with shape [",
                          inp0.SummarizeValue(10), "]."));
          out_vec(j) = 0;
        }
      }
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("ConcatOffset").Device(DEVICE_CPU),
                        ConcatOffsetOp);

REGISTER_KERNEL_BUILDER(Name("ConcatOffset")
                            .Device(DEVICE_GPU)
                            .HostMemory("concat_dim")
                            .HostMemory("shape")
                            .HostMemory("offset"),
                        ConcatOffsetOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("ConcatOffset")
                            .Device(DEVICE_SYCL)
                            .HostMemory("concat_dim")
                            .HostMemory("shape")
                            .HostMemory("offset"),
                        ConcatOffsetOp);
#endif  // TENSORFLOW_USE_SYCL

template <typename SrcT, typename DstT>
class FusedConcatCastOp : public OpKernel {
 public:
  explicit FusedConcatCastOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("Truncate", &use_truncation_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor* concat_dim_tensor;
    const char* axis_attribute_name = "axis";
    OP_REQUIRES_OK(c, c->input(axis_attribute_name, &concat_dim_tensor));
    OP_REQUIRES(c, IsLegacyScalar(concat_dim_tensor->shape()),
                errors::InvalidArgument(
                    axis_attribute_name,
                    " tensor should be a scalar integer, but got shape ",
                    concat_dim_tensor->shape().DebugString()));
    int64_t concat_dim;
    OP_REQUIRES(
        c,
        (concat_dim_tensor->dtype() == DT_INT32 ||
         concat_dim_tensor->dtype() == DT_INT64),
        errors::InvalidArgument(axis_attribute_name,
                                " tensor should be int32 or int64, but got ",
                                DataTypeString(concat_dim_tensor->dtype())));

    if (concat_dim_tensor->dtype() == DT_INT32) {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor->scalar<int32>()());
    } else {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor->scalar<int64>()());
    }

    OP_REQUIRES(
        c, use_truncation_ == false,
        errors::Unimplemented("Truncate attribute is not implemented."));

    OpInputList values;
    OP_REQUIRES_OK(c, c->input_list("values", &values));
    const int N = values.size();
    const int input_dims = values[0].dims();
    const TensorShape& input_shape = values[0].shape();

    int32 axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(c,
                (0 <= axis && axis < input_dims) ||
                    (allow_legacy_scalars() && concat_dim == 0),
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));

    int64_t inputs_flat_dim0 = 1;
    for (int d = 0; d < axis; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    int64_t output_concat_dim = 0;
    const bool input_is_scalar = IsLegacyScalar(input_shape);
    int64_t row_size = 0;
    std::vector<ptrdiff_t> sizes;
    sizes.reserve(N);
    std::vector<const SrcT*> input_pointers;
    input_pointers.reserve(N);

    for (int i = 0; i < N; ++i) {
      const auto& in = values[i];
      const bool in_is_scalar = IsLegacyScalar(in.shape());
      OP_REQUIRES(
          c, in.dims() == input_dims || (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", in.shape().DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == axis) {
          continue;
        }
        OP_REQUIRES(
            c, in.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                input_shape.DebugString(), " vs. shape[", i,
                "] = ", in.shape().DebugString()));
      }
      if (in.NumElements() > 0) {
        int64_t inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        sizes.push_back(inputs_flat_dim1);
        row_size += sizes.back();

        const SrcT* data_ptr = in.flat<SrcT>().data();
        input_pointers.push_back(data_ptr);
      }
      output_concat_dim += in.dims() > 0 ? in.dim_size(axis) : 1;
    }

    TensorShape output_shape(input_shape);
    if (output_shape.dims() == 0) {
      output_shape.AddDim(output_concat_dim);
    } else {
      output_shape.set_dim(axis, output_concat_dim);
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));

    if (output->NumElements() > 0) {
      auto output_flat = output->flat<DstT>();
      DstT* out_ptr = output_flat.data();

      auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
      int num_threads = std::min(
          static_cast<int>((output->NumElements() * sizeof(DstT)) / 4096),
          worker_threads->num_threads);
      if (num_threads == 1 || num_threads == 0) {
        for (int64_t i = 0; i < inputs_flat_dim0; ++i) {
          for (int64_t j = 0; j < N; ++j) {
            auto size = sizes[j];
#if INTEL_MKL
            castElements(out_ptr, input_pointers[j], size);
#else
            for (int64_t s = 0; s < size; ++s) {
              out_ptr[s] = static_cast<DstT>(*input_pointers[j]++);
            }
#endif  // INTEL_MKL
            out_ptr += size;
          }
        }
      } else {
        // Sharded mode.
        auto work = [&row_size, &sizes, &input_pointers, &out_ptr, &N,
                     &inputs_flat_dim0](int64_t start, int64_t end) {
          int64_t skipped_rows = start / row_size;
          DstT* out = out_ptr + skipped_rows * row_size;
          DstT* out_start = out_ptr + start;
          DstT* out_end = out_ptr + end;

          // Handle partial row at start
          if (out < out_start) {
            for (size_t i = 0; i < N; ++i) {
              ptrdiff_t size = sizes[i];
              ptrdiff_t offset = out_start - out;
              if (size <= offset) {
                out += size;
                continue;
              }

              const SrcT* data_ptr =
                  input_pointers[i] + skipped_rows * sizes[i];
              if (offset > 0) {
                out += offset;
                data_ptr += offset;
                size -= offset;
              }
              size = std::min(size, out_end - out);
              if (size <= 0) break;
#if INTEL_MKL
              castElements(out, data_ptr, size);
#else
              for (int64_t s = 0; s < size; ++s) {
                out[s] = static_cast<DstT>(*data_ptr++);
              }
#endif  // INTEL_MKL
              out += size;
            }
            ++skipped_rows;
          }
          if (out == out_end) return;
          CHECK(out >= out_start);
          CHECK(out < out_end);

          // Copy remaining data
          std::vector<const SrcT*> inp;
          inp.reserve(N);
          for (size_t i = 0; i < N; ++i) {
            const SrcT* data_ptr = input_pointers[i] + skipped_rows * sizes[i];
            inp.push_back(data_ptr);
          }

          for (int64_t i = skipped_rows; i < inputs_flat_dim0; ++i) {
            for (int64_t j = 0; j < N; ++j) {
              ptrdiff_t size = std::min(sizes[j], out_end - out);

              for (int64_t s = 0; s < size; ++s) {
                out[s] = static_cast<DstT>(*inp[j]++);
              }

              out += size;
              if (out == out_end) return;
            }
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers,
              output_flat.size(), sizeof(DstT) /* cost_per_unit */, work);
      }
    }
  }

 private:
#if INTEL_MKL
  static void castElements(Eigen::bfloat16* dst, const float* src, ptrdiff_t size) {
    dnnl::cvt_float_to_bfloat16((dnnl_bfloat16_t *)dst, (const float *)src, size);
  }

  static void castElements(float* dst, const Eigen::bfloat16* src, ptrdiff_t size) {
    dnnl::cvt_bfloat16_to_float((float *)dst, (const dnnl_bfloat16_t *)src, size);
  }
#endif  // INTEL_MKL

  bool use_truncation_;
};


#define REGISTER_CONCATCAST(src_type, dst_type)                 \
  REGISTER_KERNEL_BUILDER(Name("FusedConcatCast")               \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<src_type>("SrcT") \
                              .TypeConstraint<dst_type>("DstT") \
                              .HostMemory("axis"),              \
                          FusedConcatCastOp<src_type, dst_type>)

REGISTER_CONCATCAST(float, Eigen::bfloat16);
REGISTER_CONCATCAST(Eigen::bfloat16, float);

#undef REGISTER_CONCATCAST
}  // namespace tensorflow

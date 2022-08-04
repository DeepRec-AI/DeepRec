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

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

#include <algorithm>
#include <array>
#include <vector>

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/stream_executor/device_description.h"

namespace xla {
namespace gpu {

namespace {

// Return whether the given shape is rank 2 excluding the batch dimensions.
bool IsRank2(const Shape& shape, int64 batch_dimensions_size) {
  return shape.rank() == batch_dimensions_size + 2;
}

// In a gemm operation where output = lhs * rhs, check whether the given shapes
// are valid for the operation.
bool AreValidGemmShapes(const Shape& lhs_shape, const Shape& rhs_shape,
                        const Shape& output_shape,
                        int64 batch_dimensions_size) {
  // The inputs and the output must
  // 1) be matrices with no padding and a non-zero number of elements,
  // 2) have an allowed element type.
  PrimitiveType output_primitive_type = output_shape.element_type();
  bool type_is_allowed =
      (output_primitive_type == F16 || output_primitive_type == F32 ||
       output_primitive_type == F64 || output_primitive_type == C64 ||
       output_primitive_type == C128);
  return type_is_allowed && IsRank2(lhs_shape, batch_dimensions_size) &&
         IsRank2(rhs_shape, batch_dimensions_size) &&
         IsRank2(output_shape, batch_dimensions_size) &&
         !ShapeUtil::IsZeroElementArray(lhs_shape) &&
         !ShapeUtil::IsZeroElementArray(rhs_shape);
}

// Given a shape and a group of contiguous dimensions in the shape, returns
// a tuple of three values (major, middle, minor), where major is the size of
// the dimensions more major then the given dimensions, minor is the size of
// dimensions more minor then the given dimensions, and middle is the size of
// the given dimensions.
std::array<int64, 3> PartitionShapeByMiddleDimensions(
    const Shape& shape, absl::Span<const int64> dims_middle) {
  CHECK(LayoutUtil::AreDimensionsConsecutive(shape.layout(), dims_middle));
  std::array<int64, 3> values = {1, 1, 1};
  enum Segment { kMajor = 0, kMiddle = 1, kMinor = 2 };
  Segment cur_segment = kMinor;

  for (int64 cur_dim : LayoutUtil::MinorToMajor(shape)) {
    if (cur_segment != kMajor) {
      // Handle change of segments.
      bool cur_dim_in_middle = absl::c_linear_search(dims_middle, cur_dim);
      if (cur_segment == kMinor) {
        if (cur_dim_in_middle) {
          cur_segment = kMiddle;
        }
      } else if (cur_segment == kMiddle) {
        if (!cur_dim_in_middle) {
          cur_segment = kMajor;
        }
      }
    }
    values[cur_segment] *= shape.dimensions(cur_dim);
  }
  return values;
}

}  // namespace

bool IsMatrixMultiplication(const HloInstruction& dot) {
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  // If gemm can accept the operand shapes, use it rather than a custom
  // kernel.
  if (AreValidGemmShapes(lhs_shape, rhs_shape, dot.shape(),
                         dim_numbers.lhs_batch_dimensions_size())) {
    // The size of the reduction dimension should match. The shape inference
    // guarantees this invariant, so the check here is for programming
    // errors.
    CHECK_EQ(lhs_shape.dimensions(dim_numbers.lhs_contracting_dimensions(0)),
             rhs_shape.dimensions(dim_numbers.rhs_contracting_dimensions(0)));
    return true;
  }
  return false;
}

bool IsCublasGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kGemmCallTarget;
}

std::array<int64, 3> GetReductionTiling(
    const ReductionDimensions& reduction_dimensions,
    int smallest_input_dtype_bits,
    absl::optional<CudaComputeCapability> cuda_compute_capability) {
  if (reduction_dimensions.is_row_reduction) {
    int64 tile_z = std::min(reduction_dimensions.dimensions[0], int64{8});
    if (reduction_dimensions.dimensions[1] == 1) {
      CHECK_EQ(reduction_dimensions.dimensions[0], 1);
      return {tile_z, 1, 16};
    }
    if (reduction_dimensions.dimensions[2] % (kWarpSize * kWarpSize * 64) ==
        0) {
      return {tile_z, 1, 64};
    }
    int cc_major = 0;
    if (cuda_compute_capability) {
      cc_major = cuda_compute_capability->cc_major;
    }
    int unroll_x = 8;
    if (cc_major >= 6 && smallest_input_dtype_bits == 16) {
      unroll_x = 16;
    } else if (cc_major >= 6 && smallest_input_dtype_bits == 8) {
      unroll_x = 64;
    }
    return {tile_z, 1, unroll_x};
  }

  // Column reduction.
  return {1, 128, 1};
}

const char* const kCudnnSoftmaxCallTarget =
    "__cudnn$Softmax";

bool IsCustomCallToDnnSoftmax(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnSoftmaxCallTarget;
}

const char* const kCudnnBatchNormForwardInferenceCallTarget =
    "__cudnn$batchNormalizationForwardInference";
const char* const kCudnnBatchNormForwardTrainingCallTarget =
    "__cudnn$batchNormalizationForwardTraining";
const char* const kCudnnBatchNormBackwardCallTarget =
    "__cudnn$batchNormalizationBackward";

bool IsCustomCallToDnnBatchNorm(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnBatchNormForwardInferenceCallTarget ||
         target == kCudnnBatchNormForwardTrainingCallTarget ||
         target == kCudnnBatchNormBackwardCallTarget;
}

const char* const kGemmCallTarget = "__cublas$gemm";
const char* const kCudnnConvForwardCallTarget = "__cudnn$convForward";
const char* const kCudnnConvBackwardInputCallTarget =
    "__cudnn$convBackwardInput";
const char* const kCudnnConvBackwardFilterCallTarget =
    "__cudnn$convBackwardFilter";
const char* const kCudnnConvBiasActivationForwardCallTarget =
    "__cudnn$convBiasActivationForward";

bool IsCustomCallToDnnConvolution(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnConvForwardCallTarget ||
         target == kCudnnConvBackwardInputCallTarget ||
         target == kCudnnConvBackwardFilterCallTarget ||
         target == kCudnnConvBiasActivationForwardCallTarget;
}

const char* const kCusolverCholeskyCallTarget = "__cusolver$cholesky";

bool IsCustomCallToCusolver(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCusolverCholeskyCallTarget;
}

bool ImplementedAsLibraryCall(const HloInstruction& hlo) {
  return IsCublasGemm(hlo) || IsCustomCallToDnnBatchNorm(hlo) ||
         IsCustomCallToDnnConvolution(hlo) || IsCustomCallToDnnSoftmax(hlo);
}

bool IsReductionFromOrToContiguousDimensions(const HloInstruction& reduce) {
  if (HloOpcode::kReduce != reduce.opcode()) {
    return false;
  }

  // TODO(b/129698548): Remove this check after fixing the bug.
  if (reduce.shape().element_type() == C128) {
    return false;
  }

  const HloInstruction* input = reduce.operand(0);
  std::vector<int64> dims_to_keep;
  for (int64 dim = 0; dim < input->shape().dimensions().size(); ++dim) {
    if (!absl::c_linear_search(reduce.dimensions(), dim)) {
      dims_to_keep.push_back(dim);
    }
  }
  if (!LayoutUtil::AreDimensionsConsecutive(input->shape().layout(),
                                            dims_to_keep) &&
      !LayoutUtil::AreDimensionsConsecutive(input->shape().layout(),
                                            reduce.dimensions())) {
    return false;
  }

  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(reduce);

  if (reduction_dimensions.is_row_reduction) {
    // For row reduction, the tile block is 1 x tile_size_x, and we are reducing
    // along tile_size_x which needs to be large enough to make the tiling
    // implementation efficient.
    return reduction_dimensions.dimensions[2] >= kWarpSize;
  }

  // For column reduction, the tile block is tile_size_y x tile_size_x, and we
  // are reducing along tile_size_y. Only tile_size_y needs to be
  // large enough to make the tiling implementation efficient.
  return reduction_dimensions.dimensions[1] >= kWarpSize;
}

bool IsInputFusibleSlices(const HloInstruction& unnested_hlo,
                          bool verify_no_strides) {
  if (!unnested_hlo.IsInputFusion()) {
    return false;
  }

  auto is_non_strided = [](const std::vector<int64>& strides) -> bool {
    return absl::c_all_of(strides, [](int stride) { return stride == 1; });
  };

  const HloInstruction* root = unnested_hlo.fused_expression_root();
  if (root->opcode() == HloOpcode::kSlice) {
    return !verify_no_strides || is_non_strided(root->slice_strides());
  }

  if (root->opcode() != HloOpcode::kTuple) {
    return false;
  }

  return absl::c_all_of(root->operands(), [&](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kSlice &&
           (!verify_no_strides || is_non_strided(instr->slice_strides()));
  });
}

ReductionDimensions GetReductionKindAndContiguousComponents(
    const HloInstruction& reduce) {
  const Shape& input_shape = reduce.operand(0)->shape();
  absl::Span<const int64> dims_to_reduce = reduce.dimensions();
  DimensionVector dims_to_keep;
  for (int64 dim = 0; dim < input_shape.rank(); ++dim) {
    if (!absl::c_linear_search(dims_to_reduce, dim)) {
      dims_to_keep.push_back(dim);
    }
  }

  if (dims_to_keep.empty()) {
    return {/*is_row_reduction=*/true,
            {1, 1, ShapeUtil::ElementsIn(input_shape)}};
  }

  if (LayoutUtil::AreDimensionsConsecutive(input_shape.layout(),
                                           dims_to_keep)) {
    std::array<int64, 3> shape_partition =
        PartitionShapeByMiddleDimensions(input_shape, dims_to_keep);
    if (shape_partition[1] == 1) {
      return {/*is_row_reduction=*/true,
              {1, 1, shape_partition[0] * shape_partition[2]}};
    }
    if (shape_partition[2] == 1) {
      return {/*is_row_reduction=*/false,
              {1, shape_partition[0], shape_partition[1]}};
    }
    return {/*is_row_reduction=*/true, shape_partition};
  }

  std::array<int64, 3> shape_partition =
      PartitionShapeByMiddleDimensions(input_shape, dims_to_reduce);

  if (shape_partition[2] == 1) {
    return {/*is_row_reduction=*/true,
            {1, shape_partition[0], shape_partition[1]}};
  }
  return {/*is_row_reduction=*/false, shape_partition};
}

// This emits a device-side call to
// "i32 vprintf(i8* fmt, arguments_type* arguments)" in the driver; see
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
llvm::Value* EmitPrintf(absl::string_view fmt,
                        absl::Span<llvm::Value* const> arguments,
                        llvm::IRBuilder<>* builder) {
  std::vector<llvm::Type*> argument_types;

  // Variadic arguments implicit promotion [1] converts float to double,
  // and bool/char/short are converted to int.
  // [1] https://en.cppreference.com/w/cpp/language/variadic_arguments
  auto requires_int32_promotion = [](llvm::Type* type) {
    return type->isIntegerTy(/*BitWidth=*/1) ||
           type->isIntegerTy(/*BitWidth=*/8) ||
           type->isIntegerTy(/*BitWidth=*/16);
  };
  auto requires_double_promotion = [](llvm::Type* type) {
    return type->isFloatingPointTy();
  };

  for (auto argument : arguments) {
    llvm::Type* type = argument->getType();
    if (requires_double_promotion(type)) {
      argument_types.push_back(builder->getDoubleTy());
    } else if (requires_int32_promotion(type)) {
      argument_types.push_back(builder->getInt32Ty());
    } else {
      argument_types.push_back(type);
    }
  }
  auto* arguments_type = llvm::StructType::create(argument_types);
  llvm::Value* arguments_ptr = builder->CreateAlloca(arguments_type);
  for (size_t i = 0; i < arguments.size(); ++i) {
    llvm::Value* value = arguments[i];
    llvm::Type* type = value->getType();
    if (requires_double_promotion(type)) {
      value = builder->CreateFPCast(value, builder->getDoubleTy());
    } else if (requires_int32_promotion(type)) {
      value = builder->CreateIntCast(value, builder->getInt32Ty(),
                                     /*isSigned=*/true);
    }
    builder->CreateStore(
        value, builder->CreateGEP(arguments_ptr, {builder->getInt64(0),
                                                  builder->getInt32(i)}));
  }
  llvm::Type* ptr_ty = builder->getInt8Ty()->getPointerTo();
  return builder->CreateCall(
      builder->GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
          "vprintf",
          llvm::FunctionType::get(builder->getInt32Ty(), {ptr_ty, ptr_ty},
                                  /*isVarArg=*/false)),
      {builder->CreateGlobalStringPtr(llvm_ir::AsStringRef(fmt)),
       builder->CreatePointerCast(arguments_ptr, ptr_ty)});
}

// Helper function to emit call to AMDGPU shfl_down function.
llvm::Value* EmitAMDGPUShflDown(llvm::Value* value, llvm::Value* offset,
                                llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  CHECK_EQ(value->getType()->getPrimitiveSizeInBits(), 32);
  auto* i32_ty = b->getInt32Ty();
  llvm::FunctionCallee shfl_fn = module->getOrInsertFunction(
      llvm_ir::AsStringRef("__ockl_readuplane_i32"),
      llvm::FunctionType::get(/*Result=*/i32_ty, {i32_ty, i32_ty},
                              /*isVarArg=*/false));
  // AMDGPU device function requires first argument as i32.
  llvm::Value* result =
      b->CreateCall(shfl_fn, {b->CreateBitCast(value, i32_ty), offset});
  // AMDGPU device function always returns an i32 type.
  return b->CreateBitCast(result, value->getType());
}

// Helper function to emit call to NVPTX shfl_down intrinsic.
llvm::Value* EmitNVPTXShflDown(llvm::Value* value, llvm::Value* offset,
                               llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Intrinsic::ID llvm_intrinsic_id;
  CHECK_EQ(value->getType()->getPrimitiveSizeInBits(), 32);
  if (value->getType()->isFloatTy()) {
    llvm_intrinsic_id = llvm::Intrinsic::nvvm_shfl_sync_down_f32;
  } else {
    llvm_intrinsic_id = llvm::Intrinsic::nvvm_shfl_sync_down_i32;
  }
  llvm::Function* intrinsic =
      llvm::Intrinsic::getDeclaration(module, llvm_intrinsic_id, {});
  return b->CreateCall(
      intrinsic, {b->getInt32(-1), value, offset, b->getInt32(kWarpSize - 1)});
}

llvm::Value* EmitFullWarpShuffleDown(llvm::Value* value, llvm::Value* offset,
                                     llvm::IRBuilder<>* builder) {
  int bit_width = value->getType()->getPrimitiveSizeInBits();
  llvm::Module* module = builder->GetInsertBlock()->getModule();
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());

  // Special case for efficiency
  if (value->getType()->isFloatTy() && bit_width == 32) {
    if (target_triple.isNVPTX()) {
      return EmitNVPTXShflDown(value, offset, builder);
    } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
      return EmitAMDGPUShflDown(value, offset, builder);
    } else {
      LOG(FATAL) << "Invalid triple " << target_triple.str();
    }
  }

  // We must split values wider than 32 bits as the "shfl" instruction operates
  // on 32-bit values.
  int num_segments = CeilOfRatio(bit_width, 32);
  llvm::Value* x = builder->CreateBitCast(
      builder->CreateZExt(
          builder->CreateBitCast(value, builder->getIntNTy(bit_width)),
          builder->getIntNTy(32 * num_segments)),
      llvm::VectorType::get(builder->getInt32Ty(), num_segments));
  for (int i = 0; i < num_segments; ++i) {
    llvm::Value* insert_val;
    if (target_triple.isNVPTX()) {
      insert_val = EmitNVPTXShflDown(builder->CreateExtractElement(x, i),
                                     offset, builder);
    } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
      insert_val = EmitAMDGPUShflDown(builder->CreateExtractElement(x, i),
                                      offset, builder);
    } else {
      LOG(FATAL) << "Invalid triple " << target_triple.str();
    }
    x = builder->CreateInsertElement(x, insert_val, i);
  }
  return builder->CreateBitCast(
      builder->CreateTrunc(
          builder->CreateBitCast(x, builder->getIntNTy(32 * num_segments)),
          builder->getIntNTy(bit_width)),
      value->getType());
}

StatusOr<CudnnConvKind> GetCudnnConvKind(
    const HloCustomCallInstruction* instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kCudnnConvForwardCallTarget) {
    return CudnnConvKind::kForward;
  }
  if (target == kCudnnConvBackwardInputCallTarget) {
    return CudnnConvKind::kBackwardInput;
  }
  if (target == kCudnnConvBackwardFilterCallTarget) {
    return CudnnConvKind::kBackwardFilter;
  }
  if (target == kCudnnConvBiasActivationForwardCallTarget) {
    return CudnnConvKind::kForwardActivation;
  }
  return InternalError("Unexpected call target: %s", target);
}

StatusOr<se::dnn::ConvolutionKind> GetDnnConvolutionKind(
    const HloCustomCallInstruction* instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kCudnnConvForwardCallTarget) {
    return se::dnn::ConvolutionKind::FORWARD;
  }
  if (target == kCudnnConvBackwardInputCallTarget) {
    return se::dnn::ConvolutionKind::BACKWARD_DATA;
  }
  if (target == kCudnnConvBackwardFilterCallTarget) {
    return se::dnn::ConvolutionKind::BACKWARD_FILTER;
  }
  return InternalError("Unexpected call target: %s", target);
}

StatusOr<se::dnn::DataType> GetDnnDataType(
    const HloCustomCallInstruction* conv) {
  PrimitiveType output_primitive_type =
      conv->shape().tuple_shapes(0).element_type();
  switch (output_primitive_type) {
    case F16:
      return se::dnn::ToDataType<Eigen::half>::value;
    case F32:
      return se::dnn::ToDataType<float>::value;
    case F64:
      return se::dnn::ToDataType<double>::value;
    default:
      break;
  }
  return InternalError("Unsupported convolution datatype : %s",
                       conv->ToString());
}

string CudnnConvKindToString(CudnnConvKind kind) {
  switch (kind) {
    case CudnnConvKind::kForward:
      return "forward";
    case CudnnConvKind::kBackwardFilter:
      return "backward_filter";
    case CudnnConvKind::kBackwardInput:
      return "backward_input";
    case CudnnConvKind::kForwardActivation:
      return "forward with activation";
  }
}

llvm::Value* IsBlock0Thread0(llvm::IRBuilder<>* b) {
  return b->CreateAnd(
      b->CreateICmpEQ(
          b->getInt32(0),
          EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b)),
      b->CreateICmpEQ(
          b->getInt32(0),
          EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b)));
}

bool AreFusedReductionOutputsConsistent(
    absl::Span<const HloInstruction* const> output_instructions,
    const HloInstruction* first_reduce) {
  for (const HloInstruction* inst : output_instructions) {
    if (IsReductionFromOrToContiguousDimensions(*inst)) {
      // Shapes, layouts and dimensions must be the same for all reduces
      // inside of this fusion.
      // TODO(tjoerg): Relax the shape constraint. The datatype does not matter.
      if (!(ShapeUtil::Equal(first_reduce->shape(), inst->shape()) &&
            ShapeUtil::Equal(first_reduce->operand(0)->shape(),
                             inst->operand(0)->shape()) &&
            ShapeUtil::Equal(first_reduce->operand(1)->shape(),
                             inst->operand(1)->shape()) &&
            first_reduce->dimensions() == inst->dimensions())) {
        return false;
      }
    } else {
      if (!(ShapeUtil::CompatibleIgnoringElementType(
                first_reduce->operand(0)->shape(), inst->shape()) &&
            LayoutUtil::Equal(first_reduce->operand(0)->shape().layout(),
                              inst->shape().layout()))) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace gpu
}  // namespace xla

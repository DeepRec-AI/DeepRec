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

#include "tensorflow/compiler/xla/service/gpu/thunk_emitter.h"

#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/async_out_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_softmax_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"
#endif

namespace xla {
namespace gpu {

std::unique_ptr<Thunk> ThunkEmitter::BuildFftThunk(const HloInstruction* inst) {
  const HloInstruction* operand = inst->operand(0);
  return absl::make_unique<FftThunk>(
      inst->fft_type(), inst->fft_length(),
      /*input_buffer=*/GetAllocationSlice(*operand),
      /*output_buffer=*/GetAllocationSlice(*inst),
      /*input_shape=*/operand->shape(),
      /*output_shape=*/inst->shape(), inst);
}

std::unique_ptr<Thunk> ThunkEmitter::BuildTriangularSolveThunk(
    const HloInstruction* inst) {
  const HloInstruction* a = inst->operand(0);
  const HloInstruction* b = inst->operand(1);
  int64 m = b->shape().dimensions(b->shape().rank() - 2);
  int64 n = b->shape().dimensions(b->shape().rank() - 1);
  int64 batch_size = std::accumulate(
      b->shape().dimensions().begin(), b->shape().dimensions().end() - 2,
      int64{1}, [](int64 a, int64 b) { return a * b; });
  int64 elem_size =
      ShapeUtil::ByteSizeOfPrimitiveType(inst->shape().element_type());
  int64 a_batch_stride = inst->triangular_solve_options().left_side()
                             ? m * m * elem_size
                             : n * n * elem_size;
  int64 b_batch_stride = m * n * elem_size;
  return absl::make_unique<TriangularSolveThunk>(
      inst->triangular_solve_options(),
      /*a_input_buffer=*/GetAllocationSlice(*a),
      /*b_input_buffer=*/GetAllocationSlice(*inst),
      inst->shape().element_type(), batch_size, m, n, a_batch_stride,
      b_batch_stride, inst);
}

std::unique_ptr<Thunk> ThunkEmitter::BuildGemmThunk(
    const HloInstruction* inst) {
  auto config_or = inst->backend_config<GemmBackendConfig>();
  GemmBackendConfig gemm_config = std::move(config_or.ValueOrDie());
  const HloInstruction* lhs = inst->operand(0);
  const HloInstruction* rhs = inst->operand(1);

  // The bias is passed inside the output buffer. If those buffers are shared
  // we can just use it, otherwise copy the bias values into the output buffer
  // first.
  if (gemm_config.beta() != 0.0) {
    const HloInstruction* bias = inst->operand(2);
    CHECK_EQ(bias->shape(), inst->shape());
    if (GetAllocationSlice(*bias) != GetAllocationSlice(*inst)) {
      std::vector<std::unique_ptr<Thunk>> thunks;
      thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
          /*source_buffer=*/GetAllocationSlice(*bias),
          /*destination_buffer=*/GetAllocationSlice(*inst),
          /*mem_size=*/ShapeUtil::ByteSizeOf(inst->shape()), nullptr));
      thunks.push_back(absl::make_unique<GemmThunk>(
          GetAllocationSlice(*lhs),   // The buffer assigned to LHS.
          GetAllocationSlice(*rhs),   // The buffer assigned to RHS.
          GetAllocationSlice(*inst),  // The output buffer.
          /*implements_whole_instruction=*/false, inst,
          std::move(gemm_config)));
      return absl::make_unique<SequentialThunk>(std::move(thunks), inst);
    }
  }

  return absl::make_unique<GemmThunk>(
      GetAllocationSlice(*lhs),   // The buffer assigned to LHS.
      GetAllocationSlice(*rhs),   // The buffer assigned to RHS.
      GetAllocationSlice(*inst),  // The output buffer.
      /*implements_whole_instruction=*/true, inst, std::move(gemm_config));
}

std::unique_ptr<Thunk> ThunkEmitter::BuildInfeedThunk(
    const HloInstruction* inst) {
  CHECK_EQ(HloOpcode::kInfeed, inst->opcode());

  ShapeTree<BufferAllocation::Slice> slices(inst->shape());
  slices.ForEachMutableElement(
      [&](const ShapeIndex& index, BufferAllocation::Slice* slice) {
        *slice = GetAllocationSlice(*inst, index);
      });
  return absl::make_unique<InfeedThunk>(slices, inst);
}

std::unique_ptr<Thunk> ThunkEmitter::BuildOutfeedThunk(
    const HloInstruction* inst) {
  CHECK_EQ(HloOpcode::kOutfeed, inst->opcode());

  ShapeTree<BufferAllocation::Slice> slices(inst->operand(0)->shape());
  slices.ForEachMutableElement([&](const ShapeIndex& index,
                                   BufferAllocation::Slice* slice) {
    auto status_or_slice = MaybeGetAllocationSlice(*inst->operand(0), index);
    if (status_or_slice.ok()) {
      *slice = status_or_slice.ValueOrDie();
    }
  });
  return absl::make_unique<OutfeedThunk>(std::move(slices), inst);
}

std::unique_ptr<Thunk> ThunkEmitter::BuildAsyncOutSendThunk(
    const HloInstruction* inst) {
  CHECK_EQ(HloOpcode::kAsyncOutSend, inst->opcode());

  const HloInstruction* operand = inst->operand(0);
  return absl::make_unique<AsyncOutSendThunk>(
      /*input_buffer=*/GetAllocationSlice(*operand), inst,
      inst->async_out_send_shape(), inst->rendezvous_key());
}

Status ThunkEmitter::HandleCustomCall(HloInstruction* custom_call) {
  // A CustomCall on the GPU backend can either be a custom-call to a
  // user-supplied kernel, or a call into a library like cudnn.

  // Lower custom-call to cudnn softmax op to specialized thunk.  It's part
  // of the contract of the cudnn softmax call that the 
  // feature_index and the log operands be constants.
  if (custom_call->custom_call_target() ==
      kCudnnSoftmaxCallTarget) {
    const HloInstruction* feature_index = custom_call->operand(1);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    const HloInstruction* log = custom_call->operand(2);
    CHECK(log->IsConstant());
    bool log_value = log->literal().Get<bool>({});

    AddThunkToThunkSequence(
        absl::make_unique<CudnnSoftmaxThunk>(
            /*operand=*/GetAllocationSlice(*custom_call->operand(0)),
            /*feature_index=*/feature_index_value,
            /*log=*/log_value,
            /*output=*/GetAllocationSlice(*custom_call),
            /*hlo=*/custom_call));
    return Status::OK();
  }

  // Lower custom-calls to cudnn batchnorm ops to specialized thunks.  It's part
  // of the contract of these cudnn batchnorm calls that the epsilon and
  // feature_index operands be constants.
  if (custom_call->custom_call_target() ==
      kCudnnBatchNormForwardInferenceCallTarget) {
    const HloInstruction* epsilon = custom_call->operand(5);
    CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index = custom_call->operand(6);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    AddThunkToThunkSequence(
        absl::make_unique<CudnnBatchNormForwardInferenceThunk>(
            /*operand=*/GetAllocationSlice(*custom_call->operand(0)),
            /*scale=*/GetAllocationSlice(*custom_call->operand(1)),
            /*offset=*/GetAllocationSlice(*custom_call->operand(2)),
            /*mean=*/GetAllocationSlice(*custom_call->operand(3)),
            /*variance=*/GetAllocationSlice(*custom_call->operand(4)),
            /*epsilon=*/epsilon_value,
            /*feature_index=*/feature_index_value,
            /*output=*/GetAllocationSlice(*custom_call),
            /*hlo=*/custom_call));
    return Status::OK();
  }

  auto get_batch_norm_operand_slices = [&](const HloInstruction* batch_norm) {
    std::vector<BufferAllocation::Slice> operand_slices;
    // The last 2 operands in the custom call are epsilon
    // and feature_index, so no allocation slice.
    auto num_inputs_slices = batch_norm->operand_count() - 2;
    operand_slices.reserve(num_inputs_slices);
    for (int id = 0; id < num_inputs_slices; id++) {
      operand_slices.push_back(GetAllocationSlice(*batch_norm->operand(id)));
    }
    return operand_slices;
  };

  auto get_batch_norm_output_slices = [&](const HloInstruction* batch_norm) {
    auto num_outputs = batch_norm->shape().tuple_shapes_size();
    std::vector<BufferAllocation::Slice> output_slices;
    output_slices.reserve(num_outputs);
    for (int index = 0; index < num_outputs; index++) {
      output_slices.push_back(GetAllocationSlice(*batch_norm, {index}));
    }
    return output_slices;
  };

  if (custom_call->custom_call_target() ==
      kCudnnBatchNormForwardTrainingCallTarget) {
    bool has_side_input = custom_call->operand_count() == 6;
    int epsilon_dim = (has_side_input) ? 4 : 3;
    int feature_index_dim = epsilon_dim + 1;
    const HloInstruction* epsilon = custom_call->operand(epsilon_dim);
    CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index =
        custom_call->operand(feature_index_dim);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    std::vector<BufferAllocation::Slice> operand_slices =
        get_batch_norm_operand_slices(custom_call);
    std::vector<BufferAllocation::Slice> output_slices =
        get_batch_norm_output_slices(custom_call);
    // If batchnorm does not have a reserve space and workspace, the number of
    // outputs will be 3.
    if (custom_call->shape().tuple_shapes_size() > 3) {
      VLOG(1) << "BatchNorm forward reserve space buffer slice: "
              << output_slices[3].ToString();
    }

    AddThunkToThunkSequence(
        absl::make_unique<CudnnBatchNormForwardTrainingThunk>(
            /*operands=*/std::move(operand_slices),
            /*outputs=*/std::move(output_slices),
            /*epsilon=*/epsilon_value,
            /*feature_index=*/feature_index_value,
            /*output_tuple=*/GetAllocationSlice(*custom_call),
            /*hlo=*/custom_call));
    return Status::OK();
  }

  if (custom_call->custom_call_target() == kCudnnBatchNormBackwardCallTarget) {
    bool use_reserve_space = custom_call->operand_count() == 8;
    int epsilon_dim = (use_reserve_space) ? 6 : 5;
    int feature_index_dim = epsilon_dim + 1;
    const HloInstruction* epsilon = custom_call->operand(epsilon_dim);
    CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index =
        custom_call->operand(feature_index_dim);
    CHECK(feature_index->IsConstant());
    int64 feature_index_value = feature_index->literal().Get<int64>({});

    // BatchNormGrad returns a tuple of three elements: grad_data, grad_scale,
    // grad_offset.
    std::vector<BufferAllocation::Slice> operand_slices =
        get_batch_norm_operand_slices(custom_call);
    std::vector<BufferAllocation::Slice> output_slices =
        get_batch_norm_output_slices(custom_call);
    // When BN-Grad is in a separate cluster, the argument for reserve space is
    // converted to F32 at entry. This causes thunk_emitter to request for a
    // slice 4 times bigger than what is required and allocated by BN-Forward
    // (reserve space in bn-fwd is say U8{N} while the reserve space
    // argument to bn-grad is F32{N} => num_bytes requested is 4N). Scaling down
    // the number of bytes requested in BN-Grad by a factor of num_bytes in
    // data-type (4 bytes for F32).
    if (use_reserve_space) {
      VLOG(2)
          << "BatchNorm backward reserve space buffer slice before correction: "
          << operand_slices[5].ToString();
      auto bytes_size_reserve_space_type = ShapeUtil::ByteSizeOfPrimitiveType(
          custom_call->operand(5)->shape().element_type());
      auto actual_reserve_space_size =
          operand_slices[5].size() / bytes_size_reserve_space_type;
      operand_slices[5].set_size(actual_reserve_space_size);
      VLOG(1) << "BatchNorm backward reserve space buffer slice: "
              << operand_slices[5].ToString();
    }

    AddThunkToThunkSequence(absl::make_unique<CudnnBatchNormBackwardThunk>(
        /*operands=*/std::move(operand_slices),
        /*outputs=*/std::move(output_slices),
        /*epsilon=*/epsilon_value,
        /*feature_index=*/feature_index_value,
        /*output_tuple=*/GetAllocationSlice(*custom_call),
        /*hlo=*/custom_call));
    return Status::OK();
  }

  if (IsCustomCallToDnnConvolution(*custom_call)) {
    std::vector<BufferAllocation::Slice> operand_slices;
    operand_slices.reserve(custom_call->operand_count());
    for (const auto* operand : custom_call->operands()) {
      operand_slices.push_back(GetAllocationSlice(*operand));
    }
    auto tuple_result_slice = GetAllocationSlice(*custom_call);
    auto conv_result_slice = GetAllocationSlice(*custom_call, {0});
    auto scratch_slice = GetAllocationSlice(*custom_call, {1});

    AddThunkToThunkSequence(absl::make_unique<ConvolutionThunk>(
        Cast<HloCustomCallInstruction>(custom_call), std::move(operand_slices),
        conv_result_slice, scratch_slice, tuple_result_slice));
    return Status::OK();
  }

  if (IsCublasGemm(*custom_call)) {
    AddThunkToThunkSequence(BuildGemmThunk(custom_call));
    return Status::OK();
  }

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
  if (custom_call->custom_call_target() == kCusolverCholeskyCallTarget) {
    TF_ASSIGN_OR_RETURN(CholeskyOptions options,
                        custom_call->backend_config<CholeskyOptions>());

    const Shape& shape = custom_call->operand(0)->shape();
    int ndim = shape.dimensions_size();
    CHECK_GE(ndim, 2);
    int64 n = shape.dimensions(ndim - 1);

    const auto& dims = shape.dimensions();
    int64 batch_size = std::accumulate(dims.begin(), dims.end() - 2, int64{1},
                                       [](int64 a, int64 b) { return a * b; });

    auto operand_buffer = GetAllocationSlice(*custom_call->operand(0));

    auto a_buffer = GetAllocationSlice(*custom_call, {0});
    auto workspace_buffer = GetAllocationSlice(*custom_call, {1});
    auto info_buffer = GetAllocationSlice(*custom_call, {2});

    std::vector<std::unique_ptr<Thunk>> thunks;

    if (operand_buffer != a_buffer) {
      thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
          /*source_address=*/operand_buffer,
          /*destination_buffer=*/a_buffer,
          /*mem_size=*/ShapeUtil::ByteSizeOf(shape), custom_call));
    }

    thunks.push_back(absl::make_unique<CholeskyThunk>(
        options, a_buffer, workspace_buffer, info_buffer,
        custom_call->operand(0)->shape().element_type(), batch_size, n,
        custom_call));

    // Elide the sequential thunk if there's no copy.
    if (thunks.size() == 1) {
      AddThunkToThunkSequence(std::move(thunks[0]));
    } else {
      AddThunkToThunkSequence(
          absl::make_unique<SequentialThunk>(std::move(thunks), custom_call));
    }

    return Status::OK();
  }

  if (void* call_target = CustomCallTargetRegistry::Global()->Lookup(
          custom_call->custom_call_target(), std::string(platform_name()))) {
    auto get_slices_for_instr = [&](const HloInstruction* instr) {
      ShapeTree<BufferAllocation::Slice> slices(instr->shape());
      slices.ForEachMutableElement(
          [&](const ShapeIndex& index, BufferAllocation::Slice* slice) {
            StatusOr<BufferAllocation::Slice> s =
                MaybeGetAllocationSlice(*instr, index);
            if (s.ok()) {
              *slice = s.ValueOrDie();
            }
          });
      return slices;
    };
    std::vector<ShapeTree<BufferAllocation::Slice>> operand_slices;
    for (const auto* operand : custom_call->operands()) {
      operand_slices.push_back(get_slices_for_instr(operand));
    }
    ShapeTree<BufferAllocation::Slice> result_slices =
        get_slices_for_instr(custom_call);
    AddThunkToThunkSequence(absl::make_unique<CustomCallThunk>(
        call_target, std::move(operand_slices), std::move(result_slices),
        Cast<HloCustomCallInstruction>(custom_call)->opaque(), custom_call));
    return Status::OK();
  }
#endif

  return Unimplemented("No registered implementation for custom call to \"%s\"",
                       custom_call->custom_call_target());
}

Status ThunkEmitter::HandleFft(HloInstruction* fft) {
  TF_RET_CHECK(
      LayoutUtil::IsMonotonicWithDim0Major(fft->operand(0)->shape().layout()));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(fft->shape().layout()));
  AddThunkToThunkSequence(BuildFftThunk(fft));
  return Status::OK();
}

Status ThunkEmitter::HandleTriangularSolve(HloInstruction* hlo) {
  auto has_fortran_layout = [](const Layout& layout) {
    int n = layout.minor_to_major_size();
    return layout.minor_to_major(0) == n - 2 &&
           layout.minor_to_major(1) == n - 1;
  };
  TF_RET_CHECK(has_fortran_layout(hlo->operand(0)->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(hlo->operand(1)->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(hlo->shape().layout()));

  std::vector<std::unique_ptr<Thunk>> thunks;

  // Triangular solve is in-place on 'b', so copy 'b' to the output if they
  // aren't the same buffer.
  auto operand_buffer = GetAllocationSlice(*hlo->operand(1));
  auto destination_buffer = GetAllocationSlice(*hlo);
  if (operand_buffer != destination_buffer) {
    thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
        /*source_address=*/operand_buffer,
        /*destination_buffer=*/destination_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(hlo->operand(1)->shape()), hlo));
  }

  thunks.push_back(BuildTriangularSolveThunk(hlo));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(
        absl::make_unique<SequentialThunk>(std::move(thunks), hlo));
  }
  return Status::OK();
}

Status ThunkEmitter::HandleInfeed(HloInstruction* infeed) {
  AddThunkToThunkSequence(BuildInfeedThunk(infeed));
  return Status::OK();
}

Status ThunkEmitter::HandleOutfeed(HloInstruction* outfeed) {
  AddThunkToThunkSequence(BuildOutfeedThunk(outfeed));
  return Status::OK();
}

Status ThunkEmitter::HandleAsyncOutSend(HloInstruction* async_out_send) {
  AddThunkToThunkSequence(BuildAsyncOutSendThunk(async_out_send));
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla

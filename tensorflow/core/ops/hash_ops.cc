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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {

namespace {

Status ValidateVariableResourceHandle(InferenceContext* c,
                                      ShapeAndType* shape_and_type) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    shape_and_type->shape = c->UnknownShape();
    shape_and_type->dtype = DT_INVALID;
  } else {
    *shape_and_type = (*handle_data)[0];
    DataType value_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr("dtype", &value_dtype));
    if (shape_and_type->dtype != value_dtype) {
      return errors::InvalidArgument(
          "Trying to read variable with wrong dtype. "
          "Expected ",
          DataTypeString(shape_and_type->dtype), " got ",
          DataTypeString(value_dtype));
    }
  }
  return Status::OK();
}

}  // namesapce

REGISTER_OP("HashTableOp")
    .Output("hashtable: resource")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("HashTableInitializeOp")
    .Input("hashtable: resource")
    .Attr("initialized: bool")
    .Attr("concurrent_read: bool = true")
    .Attr("children: list(string) = []")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("HashTableLookupOp")
    .Input("hashtable: resource")
    .Input("keys: int64")
    .Output("ids: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("HashTableFilterOp")
    .Input("hashtable: resource")
    .Input("other_arguments: Targuments")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("block_size: int")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("HashTableLookupWithAdmitOp")
    .Input("hashtable: resource")
    .Input("keys: int64")
    .Input("admit_strategy: resource")
    .Input("freq: int32")
    .Output("ids: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("HashTableSizeOp")
    .Input("hashtable: resource")
    .Output("size: int64")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TensibleVariableOp")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      c->set_output_handle_shapes_and_types(0,
                                            std::vector<ShapeAndType>{{s, t}});

      return Status::OK();
    });

REGISTER_OP("TensibleVariableInitializeOp")
    .Input("resource: resource")
    .Input("hashtable: resource")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("factory: func")
    .Attr("initialized: bool")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("TensibleVariableIsInitializedOp")
    .Input("resource: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("TensibleVariableGather")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("default_value: dtype")
    .Attr("validate_indices: bool = true")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRankAtLeast(handle_shape_and_type.shape, 1, &unused));
      ShapeHandle params_subshape;
      TF_RETURN_IF_ERROR(
          c->Subshape(handle_shape_and_type.shape, 1, &params_subshape));
      ShapeHandle indices_shape = c->input(1);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, params_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("TensibleVariableScatterAdd")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));
      ShapeHandle var_shape = handle_shape_and_type.shape;
      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused_updates_shape;
      ShapeHandle concat;
      ShapeHandle var_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));
      return Status::OK();
    });

REGISTER_OP("TensibleVariableScatterSub")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));
      ShapeHandle var_shape = handle_shape_and_type.shape;
      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused_updates_shape;
      ShapeHandle concat;
      ShapeHandle var_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));
      return Status::OK();
    });

REGISTER_OP("TensibleVariableScatterMul")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));
      ShapeHandle var_shape = handle_shape_and_type.shape;
      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused_updates_shape;
      ShapeHandle concat;
      ShapeHandle var_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));
      return Status::OK();
    });

REGISTER_OP("TensibleVariableScatterDiv")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));
      ShapeHandle var_shape = handle_shape_and_type.shape;
      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused_updates_shape;
      ShapeHandle concat;
      ShapeHandle var_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));
      return Status::OK();
    });

REGISTER_OP("TensibleVariableScatterUpdate")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));
      ShapeHandle var_shape = handle_shape_and_type.shape;
      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused_updates_shape;
      ShapeHandle concat;
      ShapeHandle var_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));
      return Status::OK();
    });

REGISTER_OP("HashSlice")
    .Input("slicer: int32")
    .Input("keys: int64")
    .Attr("segment_size: int")
    .Output("slice_id: int32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("CopyTensor")
    .Input("src: T")
    .Output("dst: T")
    .Attr("T: type")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("HashTableDirectRestoreOp")
    .Input("keys: int64")
    .Input("ids: int64")
    .Input("exts: int64")
    .Input("hashtable: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("TensibleVariableDirectRestoreOp")
    .Input("value: T")
    .Input("resource: resource")
    .Input("hashtable: resource")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("factory: func")
    .Attr("T: type")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("HashTableMapOp")
    .Input("hashtable: resource")
    .Input("mapper_other_args: Tmapper_args")
    .Attr("mapper: func")
    .Attr("Tmapper_args: list(type) >= 0")
    .Output("keys: int64")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // output vector
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });

REGISTER_OP("HashTableSnapshotOp")
    .Input("hashtable: resource")
    .Output("keys: int64")
    .Output("ids: int64")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });

REGISTER_OP("HashTableDeleteKeyOp")
    .Input("hashtable: resource")
    .Input("keys: int64")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // input
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &s));
      return Status::OK();
    });

REGISTER_OP("ReadOnlyHashTableAdmitStrategyOp")
    .Output("strategy: resource")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BlackListHashTableAdmitStrategyOp")
    .Output("strategy: resource")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("InitBlackList")
    .Input("strategies: resource")
    .Input("fea_names: string")
    .Input("slices: int64")
    .Input("thresholds: double")
    .Input("file_names: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("BloomFilterAdmitStrategyOp")
    .Output("admit_strategy: resource")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BloomFilterIsInitializedOp")
    .Input("resource: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("BloomFilterInitializeOp")
    .Input("admit_strategy: resource")
    .Attr("min_frequency: int")
    .Attr("num_hash_func: int")
    .Attr("slice_offset: int = 0")
    .Attr("max_slice_size: int = 1")
    .Attr("dtype: {uint8, uint16, uint32}")
    .Attr("shape: shape")
    .Attr("initialized: bool")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("BloomFilterAdmitOp")
    .Input("admit_strategy: resource")
    .Input("keys: int64")
    .Input("frequencies: T")
    .Output("admit: bool")
    .Attr("T: {uint8, uint16, uint32}")
    .SetIsStateful()
    .SetShapeFn([] (InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    }); 

}  // namespace tensorflow


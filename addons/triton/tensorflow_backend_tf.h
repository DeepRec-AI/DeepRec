// Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <map>
#include <vector>

// To avoid namespace and protobuf collision between TRITON and
// TensorFlow, we keep TensorFlow interface isolated to
// tensorflow_backend_tf. We use a strict C interface to avoid any ABI
// problems since we don't know how TF is built.

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define TRITONTF_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONTF_EXPORT __attribute__((__visibility__("default")))
#else
#define TRITONTF_EXPORT
#endif

// GPU device number that indicates that no gpu is available.
#define TRITONTF_NO_GPU_DEVICE -1

// GPU device number that indicates TRITON should do nothing to control
// the device alloaction for the network and let Tensorflow handle it.
#define TRITONTF_MODEL_DEVICE -2

// Max batch size value that indicates batching is not supported.
#define TRITONTF_NO_BATCHING 0

// Error reporting. A NULL TRITONTF_Error indicates no error, otherwise
// the error is indicated by the 'msg_'.
typedef struct {
  // The error message as a null-terminated string.
  char* msg_;
} TRITONTF_Error;

// Delete an error.
TRITONTF_EXPORT void TRITONTF_ErrorDelete(TRITONTF_Error* error);

// Input or output datatype. Protobufs can't cross the TRITONTF
// boundary so need to have this non-protobuf definition.
typedef enum {
  TRITONTF_TYPE_INVALID,
  TRITONTF_TYPE_BOOL,
  TRITONTF_TYPE_UINT8,
  TRITONTF_TYPE_UINT16,
  TRITONTF_TYPE_UINT32,
  TRITONTF_TYPE_UINT64,
  TRITONTF_TYPE_INT8,
  TRITONTF_TYPE_INT16,
  TRITONTF_TYPE_INT32,
  TRITONTF_TYPE_INT64,
  TRITONTF_TYPE_FP16,
  TRITONTF_TYPE_FP32,
  TRITONTF_TYPE_FP64,
  TRITONTF_TYPE_STRING
} TRITONTF_DataType;

typedef enum {
  TRITONTF_MODE_FP32,
  TRITONTF_MODE_FP16,
  TRITONTF_MODE_INT8,
} TRITONTF_TFTRTPrecisionMode;

// Config for TF-TRT optimization if specified
typedef struct {
  bool is_dynamic_op_;
  int64_t max_batch_size_;
  int64_t max_workspace_size_bytes_;
  TRITONTF_TFTRTPrecisionMode precision_mode_;
  int64_t minimum_segment_size_;
  int64_t max_cached_engines_;
} TRITONTF_TFTRTConfig;

// A shape
typedef struct {
  // Number of dimensions in the shape
  size_t rank_;

  // The size of each dimension. -1 indicates variables-sized
  // dimension
  int64_t* dims_;
} TRITONTF_Shape;

// Information about an input or output
typedef struct {
  // Name as null-terminated string
  char* name_;

  // Name in the model itself as null-terminated string. May be null
  // if the in-model name is the same as 'name_'
  char* inmodel_name_;

  // The data-type
  TRITONTF_DataType data_type_;

  // The shape
  TRITONTF_Shape* shape_;
} TRITONTF_IO;

// List of I/O information
typedef struct tritontf_iolist_struct {
  TRITONTF_IO* io_;
  struct tritontf_iolist_struct* next_;
} TRITONTF_IOList;

//
// Tensor
//

// Opaque handle to a tensor
struct TRITONTF_Tensor;

// List of tensors
typedef struct tritontf_tensorlist_struct {
  TRITONTF_Tensor* tensor_;
  struct tritontf_tensorlist_struct* next_;
} TRITONTF_TensorList;

// Create an new tensor list. Ownership of 'tensor' passes to the
// list.
TRITONTF_EXPORT TRITONTF_TensorList* TRITONTF_TensorListNew(
    TRITONTF_Tensor* tensor, TRITONTF_TensorList* next);

// Delete a list of tensors. Any tensors contained in the list are
// also deleted.
TRITONTF_EXPORT void TRITONTF_TensorListDelete(TRITONTF_TensorList* list);


// Create a new tensor with a given name, type and shape. 'shape_dims'
// must be nullptr if shape_rank is 0. If a tensor is intended to be used as
// GPU input for model that supports GPU I/O (see TRITONTF_ModelMakeCallable),
// 'tf_gpu_id' must be the same as the model's device id. Otherwise, negative
// value should be provided. Note that a tensor may be created on CPU if
// the data type is not supported for GPU tensor.
// Return nullptr if failed to create the tensor.
TRITONTF_EXPORT TRITONTF_Tensor* TRITONTF_TensorNew(
    const char* name, TRITONTF_DataType dtype, size_t shape_rank,
    int64_t* shape_dims, int tf_gpu_id);

// Return a tensor's datatype.
TRITONTF_EXPORT TRITONTF_DataType TRITONTF_TensorDataType(TRITONTF_Tensor* tensor);

// Return the size of a tensor datatype, in bytes.
TRITONTF_EXPORT int64_t TRITONTF_TensorDataTypeByteSize(TRITONTF_Tensor* tensor);

// Return the shape of the tensor. The shape is owned by the tensor
// and should not be modified or freed by the caller.
TRITONTF_EXPORT
TRITONTF_Shape* TRITONTF_TensorShape(TRITONTF_Tensor* tensor);

// Get the base of the tensor data. Defined only for non-string
// types.. bad things might happen if called for string type tensor.
TRITONTF_EXPORT char* TRITONTF_TensorData(TRITONTF_Tensor* tensor);

// Check whether the memory type of the tensor data is GPU.
TRITONTF_EXPORT bool TRITONTF_TensorIsGPUTensor(TRITONTF_Tensor* tensor);

// Get the size, in bytes, of the tensor data. Defined only for
// non-string types.. bad things might happen if called for string
// type tensor.
TRITONTF_EXPORT size_t TRITONTF_TensorDataByteSize(TRITONTF_Tensor* tensor);

// Get a string at a specified index within a tensor. Defined only for
// string type.. bad things might happen if called for non-string type
// tensor. The returned string is owned by the Tensor and must be
// copied if the caller requires ownership. Additionally returns the
// 'length' of the string.
TRITONTF_EXPORT const char* TRITONTF_TensorString(
    TRITONTF_Tensor* tensor, size_t idx, size_t* length);

// Set a string at a specified index within a tensor. Defined only for
// string type.. bad things might happen if called for non-string type
// tensor. The provided string is copied by the tensor so the caller
// retains ownership of 'str'. 'str' may be NULL to indicate that the
// string should be set to empty. 'length' denotes the size of the
// character sequence to copy into the string within the tensor.
TRITONTF_EXPORT void TRITONTF_TensorSetString(
    TRITONTF_Tensor* tensor, size_t idx, const char* str, size_t length);

//
// Model
//

// Opaque handle to a model
struct TRITONTF_Model;

// Create a GraphDef model.
TRITONTF_EXPORT TRITONTF_Error* TRITONTF_ModelCreateFromGraphDef(
    TRITONTF_Model** tritontf_model, const char* model_name,
    const char* model_path, const int device_id, const int num_intra_threads,
    const int num_inter_threads, const bool use_per_session_threads,
    const bool has_graph_level, const int graph_level,
    const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement,
    const std::map<int, std::vector<float>>& memory_limit_mb,
    const TRITONTF_TFTRTConfig* tftrt_config, const bool auto_mixed_precision);

// Create a SavedModel model.
TRITONTF_EXPORT TRITONTF_Error* TRITONTF_ModelCreateFromSavedModel(
    TRITONTF_Model** tritontf_model, const char* model_name,
    const char* model_path, const int device_id, const int num_intra_threads,
    const int num_inter_threads, const bool use_per_session_threads,
    const char* graph_tag, const char* signature_def,
    const bool has_graph_level, const int graph_level,
    const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement,
    const std::map<int, std::vector<float>>& memory_limit_mb,
    const TRITONTF_TFTRTConfig* tftrt_config, const bool auto_mixed_precision);

// Delete a model.
TRITONTF_EXPORT void TRITONTF_ModelDelete(TRITONTF_Model* model);

// Create a Callable for the model so that the inputs will be assumed to be from
// GPU while the outputs will be produced on GPU. The Callable will assume the
// inputs are on the same TF device (vGPU) as the model session.
// Note that depending on the data type, GPU tensor may not be supported,
// in such case, the callable will expect those unsupported I/Os to be on CPU.
TRITONTF_Error* TRITONTF_ModelMakeCallable(
    TRITONTF_Model* model, const char** input_names,
    const TRITONTF_DataType* input_types, const size_t num_inputs,
    const char** output_names, const TRITONTF_DataType* output_types,
    const size_t num_outputs);

// Get information about a model inputs. The returned list is owned by
// the model and should not be modified or freed by the caller.
TRITONTF_EXPORT TRITONTF_IOList* TRITONTF_ModelInputs(TRITONTF_Model* model);

// Get information about a model outputs. The returned list is owned
// by the model and should not be modified or freed by the caller.
TRITONTF_EXPORT TRITONTF_IOList* TRITONTF_ModelOutputs(TRITONTF_Model* model);

// Run a model using the provides input tensors to produce the named
// outputs. Ownership of the 'input_tensors' is passed to the model
// and the caller must not access (or free) it after this
// call. 'output_tensors' returns the outputs in the same order as
// 'output_names'. The caller must free 'output_tensors' by calling
// TRITONTF_TensorListDelete.
TRITONTF_EXPORT TRITONTF_Error* TRITONTF_ModelRun(
    TRITONTF_Model* model, TRITONTF_TensorList* input_tensors, size_t num_outputs,
    const char** output_names, TRITONTF_TensorList** output_tensors);

// Initialize all the operations that do require initialization.
TRITONTF_EXPORT TRITONTF_Error* TRITONTF_ModelInitialize(
    TRITONTF_Model* model, size_t num_init_operations,
    const char** init_operation_names);

// Load a library and register its ops/kernels.
TRITONTF_EXPORT TRITONTF_Error* TRITONTF_LoadAndRegisterLibrary(
    const char* path);

#ifdef __cplusplus
}  // extern "C"
#endif

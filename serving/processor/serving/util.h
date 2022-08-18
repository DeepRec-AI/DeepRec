#ifndef SERVING_PROCESSOR_SERVING_UTILS_H
#define SERVING_PROCESSOR_SERVING_UTILS_H

#include "serving/processor/serving/model_message.h"
#include "serving/processor/serving/predict.pb.h"

#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"


namespace tensorflow {
namespace processor {
namespace util {

Status GetAssetFileDefs(const MetaGraphDef& meta_graph_def,
                        std::vector<AssetFileDef>* asset_file_defs);

void AddAssetsTensorsToInputs(const StringPiece export_dir,
                              const std::vector<AssetFileDef>& asset_file_defs,
                              std::vector<std::pair<string, Tensor>>* inputs);
 
Status RunOnce(const RunOptions& run_options,
               const std::vector<std::pair<string, Tensor>>& inputs,
               const std::vector<string>& output_tensor_names,
               const std::vector<string>& target_node_names,
               std::vector<Tensor>* outputs, RunMetadata* run_metadata,
               Session* session);
Status RunRestoreCheckpoint(
    bool restore_incr_checkpoint,
    const RunOptions& run_options,
    const std::string& full_ckpt_name,
    const std::string& incr_ckpt_name,
    const std::string& savedmodel_dir,
    const StringPiece restore_op_name,
    const StringPiece variable_filename_const_op_name,
    const StringPiece incr_variable_filename_const_op_name,
    const std::vector<AssetFileDef>& asset_file_defs,
    Session* session);

Status RunRestore(const RunOptions& run_options, const string& export_dir,
                  const StringPiece restore_op_name,
                  const StringPiece variable_filename_const_op_name,
                  const std::vector<AssetFileDef>& asset_file_defs,
                  Session* session);
 
bool HasMainOp(const MetaGraphDef& meta_graph_def);

Status RunMainOp(const RunOptions& run_options, const string& export_dir,
                 const MetaGraphDef& meta_graph_def,
                 const std::vector<AssetFileDef>& asset_file_defs,
                 Session* session, const string& main_op_key);

Status RunMainOp(const RunOptions& run_options, const string& export_dir,
                 const MetaGraphDef& meta_graph_def,
                 const std::vector<AssetFileDef>& asset_file_defs,
                 Session* session, const string& main_op_key,
                 std::pair<std::string, Tensor> sparse_storage_tensor);

Status RunRestore(const RunOptions& run_options,
                  const std::string& ckpt_name,
                  const std::string& savedmodel_dir,
                  const StringPiece restore_op_name,
                  const StringPiece variable_filename_const_op_name,
                  const std::vector<AssetFileDef>& asset_file_defs,
                  Session* session, bool update_sparse, int64_t latest_version,
                  std::vector<std::pair<std::string, Tensor>>& extra_tensors);

Tensor Proto2Tensor(const eas::ArrayProto& input);

eas::PredictResponse Tensor2Response(
    const processor::Request& req,
    const processor::Response& resp,
    const SignatureInfo* info);
 
} // namespace util
} // namespace processor
} // namespace tensorflow

#endif // SERVING_PROCESSOR_SERVING_UTILS_H

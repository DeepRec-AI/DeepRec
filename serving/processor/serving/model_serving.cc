#include "serving/processor/serving/model_impl.h"
#include "serving/processor/serving/model_serving.h"
#include "serving/processor/serving/model_config.h"
#include "serving/processor/serving/model_message.h"
#include "serving/processor/serving/message_coding.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace processor {

Model::Model(const std::string& model_entry)
    : model_entry_(model_entry) {
}

Model::~Model() {
  delete impl_;
}

Status Model::Init(const char* model_config) {
  ModelConfig* config = nullptr;
  auto status = ModelConfigFactory::Create(model_config, &config);
  if (!status.ok()) {
    return status;
  }

  if (!config->warmup_file_name.empty()) {
    LOG(INFO) << "User set warmup file: " << config->warmup_file_name;
  }

  parser_ = ParserFactory::GetInstance(config->serialize_protocol,
      4);
  impl_ = ModelImplFactory::Create(config);

  return impl_->Init();
}

Status Model::Predict(const void* input_data, int input_size,
    void** output_data, int* output_size) {
  Call call;
  Status status = parser_->ParseRequestFromBuf(
      input_data, input_size, call,
      impl_->GetSignatureInfo());
  if (!status.ok()) {
    return status;
  }

  status = Predict(call.request, call.response);
  if (!status.ok()) {
    return status;
  }

  status = parser_->ParseResponseToBuf(call, output_data, output_size,
                                       impl_->GetSignatureInfo());
  return status;
}

Status Model::BatchPredict(const void* input_data[], int* input_size,
    void* output_data[], int* output_size) {
  BatchCall call;
  parser_->ParseBatchRequestFromBuf(input_data, input_size, call,
                                    impl_->GetSignatureInfo());
  auto status = Predict(call.batched_request, call.batched_response);
  if (!status.ok()) {
    return status;
  }

  parser_->ParseBatchResponseToBuf(call, output_data, output_size,
                                   impl_->GetSignatureInfo());
  return Status::OK();
}

Status Model::Predict(Request& req, Response& resp) {
  return impl_->Predict(req, resp);
}

Status Model::GetServingModelInfo(void* output_data[],
                                  int* output_size) {
  ServingModelInfo model_info;
  auto status = impl_->GetServingModelInfo(model_info);
  if (!status.ok()) {
    return status;
  }
  parser_->ParseServingModelInfoToBuf(model_info, output_data, output_size);
  return Status::OK();
}

Status Model::Rollback() {
  return impl_->Rollback();
}

std::string Model::DebugString() {
  return impl_->DebugString();
}

} // processor
} // tensorflow

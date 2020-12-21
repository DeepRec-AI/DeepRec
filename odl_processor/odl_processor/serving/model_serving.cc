#include "odl_processor/serving/model_impl.h"
#include "odl_processor/serving/model_serving.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/serving/model_message.h"
#include "odl_processor/serving/message_coding.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace processor {

Model::~Model() {
  delete impl_;
}

Status Model::Init(const char* model_config) {
  ModelConfig* config = nullptr;
  auto status = ModelConfigFactory::Create(model_config, &config);
  if (!status.ok()) {
    return status;
  }

  parser_ = ParserFactory::GetInstance(config->serialize_protocol);
  impl_ = ModelImplFactory::Create(config);

  return impl_->Init();
}

Status Model::Predict(const void* input_data, int input_size,
    void** output_data, int* output_size) {
  Call call;
  parser_->ParseRequestFromBuf(input_data, input_size, call.request);
  auto status = Predict(call.request, call.response);
  if (!status.ok()) {
    return status;
  }

  parser_->ParseResponseToBuf(call.request, call.response,
    output_data, output_size);
  return Status::OK();
}

Status Model::Predict(Request& req, Response& resp) {
  return impl_->Predict(req, resp);
}

Status Model::Rollback() {
  return impl_->Rollback();
}

std::string Model::DebugString() {
  return impl_->DebugString();
}

} // processor
} // tensorflow

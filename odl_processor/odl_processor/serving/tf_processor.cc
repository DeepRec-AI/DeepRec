#include "tf_processor.h"
#include "model_serving.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "odl_processor/serving/tf_predict.pb.h"
#include "odl_processor/serving/model_message.h"
#include "odl_processor/serving/message_coding.h"

extern "C" {
void* initialize(const char* model_entry, const char* model_config,
                 int* state) {
  auto model = new tensorflow::processor::Model();
  auto status = model->Init(model_config);
  if (!status.ok()) {
    std::cerr << "[TensorFlow] Processor initialize failed"
              << ", status:" << status.error_message() << std::endl;
    *state = -1;
    return nullptr;
  }
  
  std::cout << "[TensorFlow] Processor initialize success." << std::endl;

  *state = 0;
  return model;
}

int process(void* model_buf, const void* input_data, int input_size,
            void** output_data, int* output_size) {
  auto model = static_cast<tensorflow::processor::Model*>(model_buf);
  if (input_size == 0) {
    auto model_str = model->DebugString();
    *output_data = strndup(model_str.c_str(), model_str.length());
    *output_size = model_str.length();
    return 200;
  }

  tensorflow::processor::Call call;
  tensorflow::processor::Parser parser;
  parser.ParseRequestFromProto(input_data, input_size, call.request);
  auto status = model->Predict(call.request, call.response);
  if (!status.ok()) {
    std::string errmsg = tensorflow::strings::StrCat(
        "[TensorFlow] Processor predict failed: ",
        status.error_message());
    *output_data = strndup(errmsg.c_str(), strlen(errmsg.c_str()));
    *output_size = strlen(errmsg.c_str());
    return 500;
  }
  parser.ParseResponseToProto(call.request, call.response,
      output_data, output_size);
  return 200;
}

int batch_process(void* model_buf, const void* input_data[], int* input_size,
                  void* output_data[], int* output_size) {
  // Client batch inputs into one tensor
  return process(model_buf, input_data[0], input_size[0],
      &output_data[0], output_size);
}
}

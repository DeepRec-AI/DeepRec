#include "tf_processor.h"
#include "model_serving.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "odl_processor/serving/tf_predict.pb.h"

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

  auto status = model->Predict(input_data, input_size,
      output_data, output_size);
  if (!status.ok()) {
    std::string errmsg = tensorflow::strings::StrCat(
        "[TensorFlow] Processor predict failed: ",
        status.error_message());
    *output_data = strndup(errmsg.c_str(), strlen(errmsg.c_str()));
    *output_size = strlen(errmsg.c_str());
    return 500;
  }
  return 200;
}

int batch_process(void* model_buf, const void* input_data[], int* input_size,
                  void* output_data[], int* output_size) {
  auto model = static_cast<tensorflow::processor::Model*>(model_buf);
  if (input_size == 0) {
    auto model_str = model->DebugString();
    *output_data = strndup(model_str.c_str(), model_str.length());
    *output_size = model_str.length();
    return 200;
  }

  auto status = model->BatchPredict(input_data, input_size,
      output_data, output_size);
  if (!status.ok()) {
    std::string errmsg = tensorflow::strings::StrCat(
        "[TensorFlow] Processor predict failed: ",
        status.error_message());
    *output_data = strndup(errmsg.c_str(), strlen(errmsg.c_str()));
    *output_size = strlen(errmsg.c_str());
    return 500;
  }
  return 200;
}

// TODO: EAS has a higher priority to call async interface.
//       Now we have no implementation of this async interface,
//       this will block EAS to call sync interface.
//
//typedef void (*DoneCallback)(const char* output, int output_size,
//    int64_t request_id, int finished, int error_code);
//
//void async_process(void* model_buf, const void* input_data,
//    int input_size, int64_t request_id, DoneCallback done) {
//  // TODO
//}

}

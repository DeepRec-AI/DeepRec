#ifndef SERVING_PROCESSOR_SERVING_MODEL_MESSAGE_H
#define SERVING_PROCESSOR_SERVING_MODEL_MESSAGE_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class Tensor;
class TensorInfo;

namespace processor {
struct Request {
  std::vector<std::pair<std::string, Tensor>> inputs;
  std::vector<std::string> output_tensor_names;
};

struct Response {
  std::vector<Tensor> outputs;
};

struct SignatureInfo {
  std::vector<std::string> input_key;
  std::vector<std::string> input_value_name;
  std::vector<std::string> output_key;
  std::vector<std::string> output_value_name;
  std::unordered_map<std::string, int> input_key_idx;
  std::unordered_map<std::string, int> input_value_name_idx;
  std::unordered_map<std::string, int> output_key_idx;
  std::unordered_map<std::string, int> output_value_name_idx;
};

struct Call {
  Request request;
  Response response;
};

struct BatchCall {
  std::vector<Request> request;
  std::vector<Response> response;

  Request batched_request;
  Response batched_response;
  int call_num; 

  Status BatchRequest();
  Status SplitResponse();
};

struct ServingModelInfo {
  std::string model_path;
};

} // processor
} // tensorflow

#endif // SERVING_PROCESSOR_SERVING_MODEL_MESSAGE_H


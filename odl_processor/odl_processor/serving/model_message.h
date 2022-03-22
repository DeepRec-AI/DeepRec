#ifndef TENSORFLOW_MODEL_MESSAGE_H
#define TENSORFLOW_MODEL_MESSAGE_H

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

struct Call {
  Request request;
  Response response;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_MODEL_MESSAGE_H


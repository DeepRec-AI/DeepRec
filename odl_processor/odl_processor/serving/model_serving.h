#ifndef TENSORFLOW_SERVING_MODEL_H
#define TENSORFLOW_SERVING_MODEL_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class Tensor;
namespace processor {
class ModelImpl;
class Request;
class Response;
class IParser;
class Model {
 public:
  Model(const std::string& model_entry);
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Status Init(const char* model_config);
  Status Predict(const void* input_data, int input_size,
      void** output_data, int* output_size);
  Status Predict(Request& req, Response& resp);
  
  Status BatchPredict(const void* input_data[], int* input_size,
      void* output_data[], int* output_size);

  Status Rollback();

  std::string DebugString();

 private:
  std::string model_entry_ = "";
  ModelImpl* impl_ = nullptr;
  IParser* parser_ = nullptr; // not owned
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_H


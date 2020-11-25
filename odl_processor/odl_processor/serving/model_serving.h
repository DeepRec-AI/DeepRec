#ifndef TENSORFLOW_SERVING_MODEL_H
#define TENSORFLOW_SERVING_MODEL_H

#include "tensorflow/core/lib/core/status.h"

class RunRequest;
class RunResponse;
namespace tensorflow {
namespace eas {
  class PredictRequest;
  class PredictResponse;
}
namespace processor {
class ModelImpl;
class Model {
 public:
  Model() = default;
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Status Init(const char* model_config, const char* model_dir);
  Status Predict(const eas::PredictRequest& req, eas::PredictResponse* resp);
  Status Rollback();

  std::string DebugString();

 private:
  ModelImpl* impl_ = nullptr;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_H


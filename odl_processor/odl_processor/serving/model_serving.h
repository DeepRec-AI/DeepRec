#ifndef TENSORFLOW_SERVING_MODEL_H
#define TENSORFLOW_SERVING_MODEL_H

#include "tensorflow/core/lib/core/status.h"
#include "model_config.h"

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
  Model(const char* model_config);
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Status Load(const char* model_dir);
  Status Warmup();
  
  Status Predict(const eas::PredictRequest& req, eas::PredictResponse* resp);
  Status Predict(const RunRequest& req, RunResponse* resp);

  std::string DebugString();

 private:
  ModelImpl* impl_;
  ModelConfig* config_;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_H


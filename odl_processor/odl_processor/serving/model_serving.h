#ifndef TENSORFLOW_SERVING_MODEL_H
#define TENSORFLOW_SERVING_MODEL_H

#include "tensorflow/core/lib/core/status.h"
#include "model_config.h"

namespace tensorflow {
namespace eas {
  class PredictRequest;
  class PredictResponse;
}

namespace processor {

class ModelImpl;
class Model {
 public:
  Model(const ModelConfig& config);

  Status Load(const char* model_dir);
  Status Predict(const eas::PredictRequest& req,
                 const eas::PredictResponse& resp);

 private:
  ModelImpl* impl_;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_H


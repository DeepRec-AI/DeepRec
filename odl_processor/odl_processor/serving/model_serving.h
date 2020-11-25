#ifndef TENSORFLOW_SERVING_MODEL_H
#define TENSORFLOW_SERVING_MODEL_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class Tensor;
namespace processor {
class ModelImpl;
class Model {
 public:
  Model() = default;
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Status Init(const char* model_config, const char* model_dir);
  Status Predict(const std::vector<std::pair<std::string, Tensor>>& inputs,
      const std::vector<std::string>& output_tensor_names,
      std::vector<Tensor>* outputs);
  Status Rollback();

  std::string DebugString();

 private:
  ModelImpl* impl_ = nullptr;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_H


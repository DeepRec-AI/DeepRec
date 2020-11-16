#ifndef TENSORFLOW_SERVING_MODEL_CONFIG_H
#define TENSORFLOW_SERVING_MODEL_CONFIG_H

namespace tensorflow {
namespace processor {
struct ModelConfig {
  int inter_op_parallelism_threads;
  int intra_op_parallelism_threads;
};

class ModelConfigFactory {
 public:
  ModelConfig Create(const char* model_config);
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_CONFIG_H


#ifndef TENSORFLOW_SERVING_MODEL_STORAGE_H
#define TENSORFLOW_SERVING_MODEL_STORAGE_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {
class Version;

class ModelStorage {
 public:
  Status Init(const char* root_dir) { return Status::OK(); }
  Status GetLatestVersion(Version& version) { return Status::OK();}
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_STORAGE_H


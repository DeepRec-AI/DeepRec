#ifndef TENSORFLOW_SERVING_MODEL_STORAGE_H
#define TENSORFLOW_SERVING_MODEL_STORAGE_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {
class Version;
class SparseStorage {
 public:
  void Create(const Version& version) {}
};

class ModelStorage {
 public:
  Status Init(const char* root_dir) { return Status::OK(); }
  Status GetLatestVersion(Version& version) { return Status::OK();}
  SparseStorage* GetSparseStorage(const Version& version) {
    return nullptr;
  }
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_STORAGE_H


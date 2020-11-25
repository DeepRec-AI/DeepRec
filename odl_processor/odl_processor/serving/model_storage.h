#ifndef TENSORFLOW_SERVING_MODEL_STORAGE_H
#define TENSORFLOW_SERVING_MODEL_STORAGE_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {
class Version;
class SparseStorage {
 public:
  Status Reset() { return Status::OK(); }
};

class ModelStorage {
 public:
  static ModelStorage* GetInstance() {
    static ModelStorage storage;
    return &storage;
  }

  Status Init(const char* root_dir) { return Status::OK(); }
  Status GetLatestVersion(Version& version) { return Status::OK();}

  SparseStorage* CreateSparseStorage(const Version& version) {
    return nullptr;
  }

 private:
  std::map<Version, SparseStorage*> store_;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_STORAGE_H


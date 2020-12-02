#ifndef TENSORFLOW_MODEL_STORE_MODEL_STORAGE_H
#define TENSORFLOW_MODEL_STORE_MODEL_STORAGE_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace io {
class OSSFileSystem;
}
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

  Status Init(const char* root_dir);
  Status GetLatestVersion(Version& version);

  SparseStorage* CreateSparseStorage(const Version& version);

 private:
  std::map<Version, SparseStorage*> store_;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_MODEL_STORE_MODEL_STORAGE_H


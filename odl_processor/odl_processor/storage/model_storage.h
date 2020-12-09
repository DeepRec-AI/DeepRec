#ifndef TENSORFLOW_STORAGE_MODEL_STORAGE_H
#define TENSORFLOW_STORAGE_MODEL_STORAGE_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class FileSystem;
namespace processor {
class Version;
class ModelConfig;
class SparseStorage;
class ModelStorage {
 public:
  ModelStorage(ModelConfig* config);

  Status Init(const char* root_dir);
  Status GetLatestVersion(Version& version);

  SparseStorage* CreateSparseStorage(const Version& version);

 private:
  Status GetFullModelVersion(Version& version);
  Status GetDeltaModelVersion(Version& version);

 private:
  std::string model_dir_;
  std::string delta_model_dir_;
  FileSystem* file_system_;   // not owned
  ModelConfig* model_config_; // not owned
};

} // processor
} // tensorflow

#endif // TENSORFLOW_STORAGE_MODEL_STORAGE_H


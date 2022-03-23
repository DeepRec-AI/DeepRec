#ifndef SERVING_ODL_PROCESSOR_STORAGE_MODEL_STORE_H
#define SERVING_ODL_PROCESSOR_STORAGE_MODEL_STORE_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class FileSystem;
namespace processor {
class Version;
class ModelConfig;
class ModelStore {
 public:
  ModelStore(ModelConfig* config);

  Status Init();
  Status GetLatestVersion(Version& version);

 protected:
  Status GetFullModelVersion(Version& version);
  Status GetDeltaModelVersion(Version& version);
  Status GetValidSavedModelDir(Version& version);
  Status DetectLatestCheckpointDir();

 protected:
  std::string savedmodel_dir_;
  std::string checkpoint_parent_dir_;
  std::string checkpoint_dir_;
  std::string delta_model_dir_;
  FileSystem* file_system_;   // not owned
  ModelConfig* model_config_; // not owned
};

} // processor
} // tensorflow

#endif // SERVING_ODL_PROCESSOR_STORAGE_MODEL_STORE_H


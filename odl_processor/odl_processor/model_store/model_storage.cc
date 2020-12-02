#include "odl_processor/model_store/model_storage.h"
#include "odl_processor/serving/model_version.h"

namespace tensorflow {
namespace processor {

Status ModelStorage::Init(const char* root_dir) {
  return Status::OK();
}

Status GetLatestVersion(Version& version) {
  return Status::OK();
}

SparseStorage* CreateSparseStorage(const Version& version) {
  return nullptr;
}

} // namespace processor
} // namespace tensorflow

#include "odl_processor/storage/model_storage.h"
#include "odl_processor/framework/model_version.h"

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

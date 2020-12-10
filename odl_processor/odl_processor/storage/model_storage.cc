#include "odl_processor/storage/model_storage.h"
#include "odl_processor/framework/model_version.h"
#include "odl_processor/storage/sparse_storage.h"
#include "odl_processor/serving/model_config.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/path.h"

namespace tensorflow {
namespace {
bool IsMetaFileName(const std::string& fname) {
  auto ext = io::Extension(fname);
  return ext == "meta";
}

std::pair<StringPiece, StringPiece> SplitBasename(StringPiece path) {
  path = io::Basename(path);

  auto pos = path.rfind('.');
  if (pos == StringPiece::npos)
    return std::make_pair(path, StringPiece(path.data() + path.size(), 0));
  return std::make_pair(
      StringPiece(path.data(), pos),
      StringPiece(path.data() + pos + 1, path.size() - (pos + 1)));
}

std::string ParseFullName(const std::string& fname) {
  auto base_name = io::Basename(fname);
  auto prefix = SplitBasename(fname).first;

  return io::JoinPath(io::Dirname(fname), prefix);
}

int64 ParseMetaFileName(const std::string& fname) {
  auto base_name = io::Basename(fname);
  auto pos = base_name.rfind('.');
  if (pos == StringPiece::npos)
    return 0;
  auto partial_name = StringPiece(base_name.data(), pos);
  pos = partial_name.rfind('-');
  if (pos == StringPiece::npos)
    return 0;
  
  auto id = StringPiece(partial_name.data() + pos + 1,
      partial_name.size() - (pos + 1));

  int64 ret = 0;
  strings::safe_strto64(id, &ret);
  return ret;
}
}
namespace processor {
ModelStorage::ModelStorage(ModelConfig* config) :
    model_config_(config) {
  savedmodel_dir_ = config->savedmodel_dir; 
  checkpoint_dir_ = config->checkpoint_dir;
  delta_model_dir_ = io::JoinPath(checkpoint_dir_, ".incremental_checkpoint");
}

Status ModelStorage::Init() {
  return Env::Default()->GetFileSystemForFile(savedmodel_dir_, &file_system_);
}

std::string ModelStorage::GetMetaGraphDir() {
  std::vector<string> file_names;
  auto status = file_system_->GetChildren(checkpoint_dir_, &file_names);
  if (!status.ok()) {
    return std::string();
  }

  for (auto fname : file_names) {
    if (IsMetaFileName(fname)) {
      return savedmodel_dir_;
    }
  }
  return std::string();
}

Status ModelStorage::GetLatestVersion(Version& version) {
  TF_RETURN_IF_ERROR(GetFullModelVersion(version));
  return GetDeltaModelVersion(version);
}

Status ModelStorage::GetFullModelVersion(Version& version) {
  std::vector<string> file_names;
  TF_RETURN_IF_ERROR(file_system_->GetChildren(checkpoint_dir_, &file_names));
  
  for (auto fname : file_names) {
    if (!IsMetaFileName(fname)) {
      continue;
    }
    auto v = ParseMetaFileName(fname);
    if (v > version.full_model_version) {
      version.full_model_name = ParseFullName(fname);
      version.full_model_version = v;
    }
  }
  return Status::OK();
}

Status ModelStorage::GetDeltaModelVersion(Version& version) {
  std::vector<string> file_names;
  TF_RETURN_IF_ERROR(file_system_->GetChildren(delta_model_dir_,
        &file_names));
  
  for (auto fname : file_names) {
    if (!IsMetaFileName(fname)) {
      continue;
    }
    auto v = ParseMetaFileName(fname);
    if (v > version.delta_model_version &&
        v > version.full_model_version) {
      version.delta_model_name = ParseFullName(fname);
      version.delta_model_version = v;
    }
  }
  return Status::OK();
}

SparseStorage* ModelStorage::CreateSparseStorage() {
  return new SparseStorage(model_config_);
}

} // namespace processor
} // namespace tensorflow

#include "odl_processor/framework/model_version.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/storage/model_store.h"
#include "odl_processor/storage/feature_store_mgr.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/cc/saved_model/constants.h"

namespace tensorflow {
namespace processor {
namespace {
bool IsMetaFileName(const std::string& fname) {
  auto ext = io::Extension(fname);
  return ext == "index";
}

bool IsIncrementalCkptPath(const std::string& fname) {
  auto pos = fname.find(".incremental_checkpoint");
  return pos != std::string::npos;
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

std::string ParseCkptFileName(const std::string& ckpt_dir,
    const std::string& fname) {
  auto prefix = SplitBasename(fname).first;
  return io::JoinPath(ckpt_dir, prefix);
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

} // namespace

ModelStore::ModelStore(ModelConfig* config) :
    model_config_(config) {
  savedmodel_dir_ = config->savedmodel_dir;
  checkpoint_parent_dir_ = config->checkpoint_dir;
}

Status ModelStore::Init() {
  return Env::Default()->GetFileSystemForFile(savedmodel_dir_, &file_system_);
}

Status ModelStore::GetLatestVersion(Version& version) {
  TF_RETURN_IF_ERROR(GetValidSavedModelDir(version));
  TF_RETURN_IF_ERROR(GetFullModelVersion(version));
  return GetDeltaModelVersion(version);
}

Status ModelStore::GetValidSavedModelDir(Version& version) {
  const string saved_model_pb_path =
      io::JoinPath(savedmodel_dir_, kSavedModelFilenamePb);
  const string saved_model_pbtxt_path =
      io::JoinPath(savedmodel_dir_, kSavedModelFilenamePbTxt);
  if (file_system_->FileExists(saved_model_pb_path).ok() ||
      file_system_->FileExists(saved_model_pbtxt_path).ok()) {
    version.savedmodel_dir = savedmodel_dir_;
  }
  return Status::OK();
}

Status ModelStore::GetFullModelVersion(Version& version) {
  TF_RETURN_IF_ERROR(DetectLatestCheckpointDir());
  TF_RETURN_IF_ERROR(file_system_->IsDirectory(checkpoint_dir_));

  std::vector<string> file_names;
  TF_RETURN_IF_ERROR(file_system_->GetChildren(checkpoint_dir_,
        &file_names));

  for (auto fname : file_names) {
    if (IsIncrementalCkptPath(fname) ||
        !IsMetaFileName(fname)) {
      continue;
    }

    auto v = ParseMetaFileName(fname);
    if (v > version.full_ckpt_version) {
      version.full_ckpt_name = ParseCkptFileName(checkpoint_dir_, fname);
      version.full_ckpt_version = v;
    }
  }
  return Status::OK();
}

Status ModelStore::GetDeltaModelVersion(Version& version) {
  delta_model_dir_ = io::JoinPath(checkpoint_dir_, ".incremental_checkpoint");
  TF_RETURN_IF_ERROR(file_system_->IsDirectory(delta_model_dir_));

  std::vector<string> file_names;
  TF_RETURN_IF_ERROR(file_system_->GetChildren(delta_model_dir_,
        &file_names));

  for (auto fname : file_names) {
    if (!IsMetaFileName(fname)) {
      continue;
    }
    auto v = ParseMetaFileName(fname);
    if (v > version.delta_ckpt_version &&
        v > version.full_ckpt_version) {
      version.delta_ckpt_name = ParseCkptFileName(delta_model_dir_, fname);
      version.delta_ckpt_version = v;
    }
  }
  return Status::OK();
}

Status ModelStore::DetectLatestCheckpointDir() {
  TF_RETURN_IF_ERROR(file_system_->IsDirectory(checkpoint_parent_dir_));
  std::vector<string> file_names;
  TF_RETURN_IF_ERROR(file_system_->GetMatchingPaths(
    io::JoinPath(checkpoint_parent_dir_ + "*/checkpoint"), &file_names));
  int64 latest_mtime_nsec = -1;
  for (auto fname : file_names) {
    FileStatistics stat;
    Status s = file_system_->Stat(fname, &stat);
    if (s.ok()) {
      if (stat.mtime_nsec > latest_mtime_nsec) {
        latest_mtime_nsec = stat.mtime_nsec;
        checkpoint_dir_ = fname.substr(0, fname.rfind('/'));
        checkpoint_dir_ += "/";
      }
    } else {
      LOG(WARNING) << "[Processor] Path: "<< fname
                   << " not a valid checkpoint_path, meg:" << s.ToString();
    }
  }

  LOG(WARNING) << "[Processor] checkpoint_dir: " << checkpoint_dir_;
  return Status::OK();
}

} // namespace processor
} // namespace tensorflow

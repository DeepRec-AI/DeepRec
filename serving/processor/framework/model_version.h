#ifndef SERVING_PROCESSOR_FRAMEWORK_MODEL_VERSION_H
#define SERVING_PROCESSOR_FRAMEWORK_MODEL_VERSION_H
#include <string.h>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace processor {
struct Version {
  int64 full_ckpt_version = -1;
  std::string full_ckpt_name = "";

  int64 delta_ckpt_version = -1;
  std::string delta_ckpt_name = "";

  std::string savedmodel_dir = "";

  Version() = default;
  ~Version() = default;
  Version(const Version&) = default;
  Version& operator=(const Version&) = default;

  const std::string DebugString() const {
    return tensorflow::strings::StrCat(
        "full_ckpt_version: ", std::to_string(full_ckpt_version),
        "\nfull_ckpt_name: ", full_ckpt_name,
        "\ndelta_ckpt_version: ", std::to_string(delta_ckpt_version),
        "\ndelta_ckpt_name: ", delta_ckpt_name,
        "\nsavedmodel_dir: ", savedmodel_dir);
  }

  bool IsFullModel() const {
    return delta_ckpt_name.empty();
  }

  bool CkptEmpty() const {
    return full_ckpt_name.empty();
  }

  bool SavedModelEmpty() const {
    return savedmodel_dir.empty();
  }

  friend bool operator ==(const Version& lhs, const Version& rhs) {
    return lhs.full_ckpt_version == rhs.full_ckpt_version
        && lhs.delta_ckpt_version == rhs.delta_ckpt_version;
  }

  friend bool operator !=(const Version& lhs, const Version& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator <(const Version& lhs, const Version& rhs) {
    return lhs.full_ckpt_version < rhs.full_ckpt_version ||
           (lhs.full_ckpt_version == rhs.full_ckpt_version &&
            lhs.delta_ckpt_version < rhs.delta_ckpt_version);
  }

  bool IsSameFullModel(const Version& other) const {
    return full_ckpt_version == other.full_ckpt_version;
  }
};

} // processor
} // tensorflow

#endif // SERVING_PROCESSOR_FRAMEWORK_MODEL_VERSION_H


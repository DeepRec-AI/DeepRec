#ifndef TENSORFLOW_SERVING_MODEL_VERSION_H
#define TENSORFLOW_SERVING_MODEL_VERSION_H

namespace tensorflow {
namespace processor {
struct Version {
  std::string full_model_version;
  std::string full_model_name;

  std::string delta_model_version;
  std::string delta_model_name;

  Version() = default;
  ~Version() = default;
  Version(const Version&) = default;
  Version& operator=(const Version&) = default;

  bool IsFullModel() const {
    return delta_model_name.empty();
  }

  friend bool operator ==(const Version& lhs, const Version& rhs) {
    return lhs.full_model_version == rhs.full_model_version
        && lhs.delta_model_version == rhs.delta_model_version;
  }

  friend bool operator !=(const Version& lhs, const Version& rhs) {
    return !(lhs == rhs);
  }

  bool IsSameFullModel(const Version& other) const {
    return full_model_version == other.full_model_version;
  }
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_VERSION_H


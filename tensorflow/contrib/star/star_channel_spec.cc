#include "tensorflow/contrib/star/star_channel_spec.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"


namespace tensorflow {
namespace {

Status ValidateHostPortPair(const string& host_port) {
  uint32 port;
  std::vector<string> parts = str_util::Split(host_port, ':');
  if (parts.size() != 2 || !strings::safe_strtou32(parts[1], &port) ||
      parts[0].find("/") != string::npos) {
    return errors::InvalidArgument("Could not interpret \"", host_port,
        "\" as a host-port pair.");
  }
  return Status::OK();
}  

} // namespace

Status StarChannelSpec::AddHostPortsJob(
    const string& job_id,
    const std::vector<string>& host_ports) {
  std::map<int, string> host_ports_map;
  for (size_t i = 0; i < host_ports.size(); ++i) {
    host_ports_map[i] = host_ports[i];
  }
  return AddHostPortsJob(job_id, host_ports_map);
}

Status StarChannelSpec::AddHostPortsJob(
    const string& job_id, const std::map<int, string>& host_ports) {
  if (!job_ids_.insert(job_id).second) {
    return errors::InvalidArgument(
        "Duplicate job ID in cluster specification: ", job_id);
  }
  for (const auto& id_host_port : host_ports) {
    TF_RETURN_IF_ERROR(ValidateHostPortPair(id_host_port.second));
  }
  host_ports_jobs_.emplace_back(job_id, host_ports);
  return Status::OK();
}

} // namespace tensorflow

#ifndef TENSORFLOW_CONTRIB_STAR_CHANNEL_SPEC_H_
#define TENSORFLOW_CONTRIB_STAR_CHANNEL_SPEC_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status.h"


namespace tensorflow {

class StarChannelSpec {
public:
  struct HostPortsJob {
    HostPortsJob(const std::string& job_id,
        const std::map<int, std::string>& host_ports)
        : job_id(job_id), host_ports(host_ports) {}
    const std::string job_id;
    const std::map<int, std::string> host_ports;
  };
  virtual ~StarChannelSpec() {}

  Status AddHostPortsJob(const std::string& job_id,
      const std::vector<string>& host_ports);

  Status AddHostPortsJob(const std::string& job_id,
      const std::map<int, std::string>& host_ports);

  const std::vector<HostPortsJob>& host_ports_jobs() const {
    return host_ports_jobs_;
  }

 private:
  std::vector<HostPortsJob> host_ports_jobs_;
  std::set<std::string> job_ids_;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_CHANNEL_CACHE_H_

#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_CHANNEL_CACHE_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_CHANNEL_CACHE_H_

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/lib/core/status.h"


namespace seastar {
class channel;
}

namespace tensorflow {

class StarChannelSpec;
class SeastarEngine;

class SeastarChannelCache {
 public:
  virtual ~SeastarChannelCache() {}

  virtual void ListWorkers(std::vector<std::string>* workers) const = 0;
  virtual void ListWorkersInJob(const string& job_name,
                                std::vector<string>* workers) = 0;
  virtual seastar::channel* FindWorkerChannel(const std::string& target) = 0;
  virtual std::string TranslateTask(const std::string& task) = 0;
};

SeastarChannelCache* NewSeastarChannelCache(
    SeastarEngine* engine, const StarChannelSpec& channel_spec);

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_CHANNEL_CACHE_H_

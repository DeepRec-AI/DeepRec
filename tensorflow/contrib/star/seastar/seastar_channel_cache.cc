#include "seastar/core/channel.hh"
#include "tensorflow/contrib/star/seastar/seastar_channel_cache.h"
#include "tensorflow/contrib/star/seastar/seastar_engine.h"
#include "tensorflow/contrib/star/star_channel_spec.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"


namespace tensorflow {
namespace {

string MakeAddress(const string& job, int task) {
  return strings::StrCat("/job:", job, "/replica:0/task:", task);
}

class CachingSeastarChannelCache : public SeastarChannelCache {
 public:
  CachingSeastarChannelCache() {}
  ~CachingSeastarChannelCache() override {}

  seastar::channel* FindWorkerChannel(const string& target) override {
    seastar::channel* ch = nullptr;
    {
      mutex_lock l(mu_);
      ch = gtl::FindPtrOrNull(channels_, target);
      if (ch) {
        return ch;
      }
    }
    ch = FindChannelOnce(target);
    if (ch) {
      mutex_lock l(mu_);
      channels_.insert({target, ch});
    }
    return ch;
  }

 protected:
  virtual seastar::channel* FindChannelOnce(const string& target) = 0;

 private:
  mutex mu_;
  std::unordered_map<string, seastar::channel*> channels_ TF_GUARDED_BY(mu_);
};

class MultiSeastarChannelCache : public CachingSeastarChannelCache {
 public:
  explicit MultiSeastarChannelCache(
      const std::vector<SeastarChannelCache*> caches)
      : CachingSeastarChannelCache(), caches_(caches) {}

  ~MultiSeastarChannelCache() override {
    for (SeastarChannelCache* cache : caches_) {
      delete cache;
    }
  }

  void ListWorkers(std::vector<string>* workers) const override {
    for (SeastarChannelCache* cache : caches_) {
      cache->ListWorkers(workers);
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) override {
    for (SeastarChannelCache* cache : caches_) {
      cache->ListWorkersInJob(job_name, workers);
    }
  }

  string TranslateTask(const string& target) override {
    mutex_lock l(mu_);
    SeastarChannelCache* cache = gtl::FindPtrOrNull(target_caches_, target);
    if (cache == nullptr) {
      for (SeastarChannelCache* c : caches_) {
        string r = c->TranslateTask(target);
        if (!r.empty()) {
          target_caches_.insert({target, c});
          cache = c;
          break;
        }
      }
    }
    CHECK(cache) << "Could not find SeastarChannelCache holding channel for "
                 << target;
    return cache->TranslateTask(target);
  }

 protected:
  seastar::channel* FindChannelOnce(const string& target) override {
    for (SeastarChannelCache* cache : caches_) {
      seastar::channel* ch(cache->FindWorkerChannel(target));
      if (ch) {
        mutex_lock l(mu_);
        target_caches_.insert({target, cache});
        return ch;
      }
    }
    return nullptr;
  }

 private:
  const std::vector<SeastarChannelCache*> caches_;
  mutex mu_;
  std::unordered_map<string, SeastarChannelCache*> target_caches_
      TF_GUARDED_BY(mu_);
};

class SparseSeastarChannelCache : public CachingSeastarChannelCache {
 public:
  SparseSeastarChannelCache(const string& job_id,
                            const std::map<int, string>& address_names,
                            SeastarEngine* engine)
      : job_id_(job_id), address_names_(address_names), engine_(engine) {
    LOG(INFO) << "Initialize SeastarChannelCache for job " << ToString();
  }

  void ListWorkers(std::vector<string>* workers) const override {
    workers->reserve(workers->size() + address_names_.size());
    for (const auto& id_host_port : address_names_) {
      workers->emplace_back(MakeAddress(job_id_, id_host_port.first));
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) override {
    if (job_name == job_id_) {
      ListWorkers(workers);
    }
  }

  string TranslateTask(const string& target) override {
    DeviceNameUtils::ParsedName parsed;
    if (!DeviceNameUtils::ParseFullName(target, &parsed)) {
      LOG(WARNING) << "Invalid target:" << target;
      return "";
    }

    if (!parsed.has_job || parsed.job != job_id_) {
      return "";
    }
    if (!parsed.has_replica || parsed.replica != 0) {
      LOG(WARNING) << "Replica ID must be 0 in target: " << target;
      return "";
    }
    int32 task = parsed.has_task ? parsed.task : -1;
    auto iter = address_names_.find(task);
    if (iter == address_names_.end()) {
      LOG(WARNING) << "Task " << task << " was not defined in sparse job "
                   << job_id_ << ": " << target;
      return "";
    }
    return iter->second;
  }

 protected:
  seastar::channel* FindChannelOnce(const string& target) override {
    const string host_port = TranslateTask(target);
    if (host_port.empty()) {
      return nullptr;
    }

    return engine_->GetChannel(host_port);
  }

 private:
  string ToString() {
    std::vector<string> task_strings;
    task_strings.reserve(address_names_.size());
    for (const auto& id_host_port : address_names_) {
      task_strings.emplace_back(
          strings::StrCat(id_host_port.first, " -> ", id_host_port.second));
    }
    return strings::StrCat(job_id_, " -> {", str_util::Join(task_strings, ", "),
                           "}");
  }

  const string job_id_;
  const std::map<int, std::string> address_names_;
  SeastarEngine* engine_;
};

} // namespace

SeastarChannelCache* NewSeastarChannelCache(SeastarEngine* engine,
                                            const StarChannelSpec& spec) {
  const int num_jobs = spec.host_ports_jobs().size();
  if (!num_jobs) {
    LOG(ERROR) << "Empty channel spec.";
    return nullptr;
  }

  std::vector<SeastarChannelCache*> caches;
  caches.reserve(num_jobs);
  for (auto& job : spec.host_ports_jobs()) {
    caches.push_back(
        new SparseSeastarChannelCache(job.job_id, job.host_ports, engine));
  }
  return caches.size() == 1 ? caches[0] : new MultiSeastarChannelCache(caches);
}

} // namespace tensorflow

#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_ENGINE_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_ENGINE_H_

#include <string>
#include <vector>
#include <map>
#include <thread>

#include "seastar/core/app-template.hh"
#include "seastar/core/distributed.hh"
#include "tensorflow/core/platform/macros.h"


namespace seastar {
class channel;
} // namespace seastar

namespace tensorflow {

class SeastarClient;
class SeastarServer;
class StarWorkerService;
class SeastarTagFactory;

using namespace seastar;

class SeastarEngine {
public:
  //protect seastar_engine only initialize once
  //seastar only support one app run
  template<typename... Args>
  static SeastarEngine* GetInstance(Args... args) {
    static SeastarEngine engine(std::forward<Args>(args)...);
    return &engine;
  }
  
  SeastarEngine(const std::string& cpuset,
                uint16_t server_number,
                uint16_t local,
                const std::string& job_name,
                StarWorkerService* worker_service);
  
  SeastarEngine(uint16_t server_number,
                uint16_t local,
                const std::string& job_name,
                StarWorkerService* worker_service);

  virtual ~SeastarEngine();

  seastar::channel* GetChannel(const std::string& server_ip);
private:
  void AsyncStart();
  void ConstructArgs(int* argc, char*** argv);
  void GetCpuset(char**);

private:
  seastar::distributed<SeastarServer> _server;
  SeastarClient* _client;
  SeastarTagFactory* _tag_factory;

  std::thread _thread;
  std::string _cpuset;
  uint16_t _local;
  std::atomic_size_t _core_id;
  std::atomic<bool> _init_ready;
  size_t _core_number;
  std::string _job_name;

  TF_DISALLOW_COPY_AND_ASSIGN(SeastarEngine);
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_ENGINE_H_

#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>

#include "tensorflow/contrib/star/seastar/seastar_client.h"
#include "tensorflow/contrib/star/seastar/seastar_cpuset.h"
#include "tensorflow/contrib/star/seastar/seastar_header.h"
#include "tensorflow/contrib/star/seastar/seastar_engine.h"
#include "tensorflow/contrib/star/seastar/seastar_server.h"
#include "tensorflow/contrib/star/seastar/seastar_tag_factory.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {
  const static int WORKER_DEFAULT_CORE_NUM = 16;
  const static int PS_DEFAULT_CORE_NUM = 16;
  const static int kWaitTimeInUs = 50000;

  // valid ipv4
  bool IsValidIp(const string& ip_string) {
    struct sockaddr_in sa;
    return (inet_pton(AF_INET, ip_string.c_str(), &(sa.sin_addr)) == 1);
  }

  // sesatar can't use hostname directly
  string HostNameToIp(const string& ip_string) {
    const auto& vec = str_util::Split(ip_string, ':');
    if (vec.size() != 2) {
      LOG(FATAL) << "Error ip:port or hostname:port info: " << ip_string;
    }

    // valid ip
    if (IsValidIp(vec[0])) {
      return ip_string;
    }

    addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;     /* ipv4 or ipv6 */
    hints.ai_socktype = SOCK_STREAM; /* stream socket */
    hints.ai_flags = AI_PASSIVE;     /* for wildcard IP address */

    addrinfo* addrs = nullptr;
    int s = getaddrinfo(vec[0].c_str(), nullptr, &hints, &addrs);
    if (s != 0) {
      LOG(ERROR) << "getaddrinfo failure, error:" << gai_strerror(s);
    }

    /*
       There are several reasons why the linked list may have more than
       one addrinfo structure, including: the network host is
       multihomed, accessible over multiple protocols (e.g., both
       AF_INET and AF_INET6); or the same service is available from
       multiple socket types (one SOCK_STREAM address and another
       SOCK_DGRAM address, for example).
     */
    std::vector<std::string> ret_ip;
    for (auto rp = addrs; rp != nullptr; rp = rp->ai_next) {
      auto addr = rp->ai_addr;
      char ip[128];
      switch(addr->sa_family) {
        case AF_INET:
          inet_ntop(AF_INET, &(((struct sockaddr_in *)addr)->sin_addr),
              ip, 128);
          LOG(INFO) << "IpV4 address:" << ip;
          break;
        case AF_INET6:
          inet_ntop(AF_INET6, &(((struct sockaddr_in6 *)addr)->sin6_addr),
              ip, 128);
          LOG(INFO) << "IpV6 address:" << ip;
          break;
        default:
          LOG(INFO) << "Invalid address.";
      }
      ret_ip.emplace_back(ip);
    }
    freeaddrinfo(addrs);
    // Here return only first address.
    return strings::StrCat(ret_ip[0], ":", vec[1]);;
  }

  bool IsWorker(const std::string& name) {
    return name == "worker" || name == "chief";
  }

  size_t GetCoreNumber(const std::string& job_name, size_t total_connections) {
    if (total_connections == 0) {
      // NOTE(rangeng.llb): allocate one core to make seastar work correctly.
      return 1;
    }

    Status s;
    int64 max_core_number = WORKER_DEFAULT_CORE_NUM;
    if (job_name == "ps") {
      s = ReadInt64FromEnvVar("NETWORK_PS_CORE_NUMBER", PS_DEFAULT_CORE_NUM,
                              &max_core_number);
    } else if (IsWorker(job_name)) {
      s = ReadInt64FromEnvVar("NETWORK_WORKER_CORE_NUMBER",
                              WORKER_DEFAULT_CORE_NUM, &max_core_number);
    } 
    if (!s.ok()) {
      LOG(WARNING) << "Please setup NETWORK_PS_CORE_NUMBER or"
                   << " NETWORK_WORKER_CORE_NUMBER";
    }
    return std::min(total_connections, (size_t)max_core_number);
  }

  // force to avoid cleanup static variables and global variables in seastar
  // engine which would trigger core
  void SeastarExit(int status, void* arg) {
    _exit(status);
  }

  // By default, ps & worker disable pin core.
  bool DisablePinCores(const std::string& job_name) {
    auto env_name = IsWorker(job_name) ?
                    "WORKER_DISABLE_PIN_CORES" :
                    "PS_DISABLE_PIN_CORES";
    bool disable_pin_core = true;
    auto status = ReadBoolFromEnvVar(env_name, true, &disable_pin_core);
    if (!status.ok()) {
      LOG(WARNING) << "Fail to get bool value: " << env_name
                   << " from env. Error mgr: " << status.error_message();
    }
    return disable_pin_core;
  }

  // By default, ps & worker disable polling.
  bool EnablePolling(const std::string& job_name) {
    auto env_name = IsWorker(job_name) ?
                    "WORKER_ENABLE_POLLING" :
                    "PS_ENABLE_POLLING";
    bool enable_poll = false;
    auto status = ReadBoolFromEnvVar(env_name, false, &enable_poll);
    if (!status.ok()) {
      LOG(WARNING) << "Fail to get bool value: " << env_name
                   << " from env. Error msg: " << status.error_message();
    }
    return enable_poll;
  }

  void ConnectAsync(seastar::channel* ch, string s, int core_id,
                    SeastarTagFactory* tag_factory, SeastarClient* client,
                    void (channel::*done)()) {
    alien::submit_to(core_id, [core_id, s, ch, tag_factory, client, done] {
      LOG(INFO) << "client start connect core:" << core_id
                << ", connect server:" << s;
      client->start(seastar::ipv4_addr{s}, s, ch, tag_factory);
      if (done != nullptr) {
        (ch->*done)();
      }
      return seastar::make_ready_future();
    });
  }
}

SeastarEngine::SeastarEngine(const std::string& cpuset, uint16_t server_number,
                             uint16_t local, const std::string& job_name,
                             StarWorkerService* worker_service) 
  : _cpuset(cpuset), _local(local), _core_id(0), _init_ready(false),
    _job_name(job_name) {
    assert(worker_service != nullptr);
    ::on_exit(SeastarExit, (void*)nullptr);
    _tag_factory = new SeastarTagFactory(worker_service);
    _core_number = GetCoreNumber(_job_name, server_number);
    _client = new SeastarClient();
    _thread = std::thread(&SeastarEngine::AsyncStart, this);
}

SeastarEngine::SeastarEngine(uint16_t server_number, uint16_t local,
                             const std::string& job_name,
                             StarWorkerService* worker_service)
  : _local(local), _core_id(0), _init_ready(false), _job_name(job_name) {
    assert(worker_service != nullptr);
    ::on_exit(SeastarExit, (void*)nullptr);
    _tag_factory = new SeastarTagFactory(worker_service);
    _core_number = GetCoreNumber(_job_name, server_number);
    _client = new SeastarClient();
    _thread = std::thread(&SeastarEngine::AsyncStart, this);
}

SeastarEngine::~SeastarEngine() { 
  // TODO: Seastar engine is static object, exit directly, need more elegant exit
  _exit(0);
}

seastar::channel* SeastarEngine::GetChannel(const std::string& server_ip) {
  size_t core_id = _core_id ++ % _core_number;
  string s = HostNameToIp(server_ip);
  auto ch = new seastar::channel(s);

  ch->set_channel_reconnect_func(std::bind(&ConnectAsync, ch, s, core_id,
        _tag_factory, _client, std::placeholders::_1));

  int retry = 200; //10 second
  while (!_init_ready.load(std::memory_order_relaxed) && retry > 0) {
    retry --;
    usleep(kWaitTimeInUs);
  }

  if (!_init_ready.load(std::memory_order_relaxed)) {
    LOG(FATAL) << "Grpc++ initialization failure, trying to connect:"<< s
               << " timeout!";
  }

  ConnectAsync(ch, s, core_id, _tag_factory, _client, nullptr);
  return ch; 
}

void SeastarEngine::GetCpuset(char** av) {
  if (_cpuset.empty()) {
    CpusetAllocator cpuset_alloc;
    _cpuset = cpuset_alloc.GetCpuset(_core_number);
  }
  if (_cpuset.empty()) {
    LOG(ERROR) << "internal error when launch grpc++ protocol, Please try other protocal";
    exit(-1);
  }

  *av = new char[_cpuset.size() + 1]();
  memcpy(*av, _cpuset.c_str(), _cpuset.size());
}

void SeastarEngine::ConstructArgs(int* argc, char*** argv) {
  *argc = 3;

  // Set av0.
  char* av0 = new char[sizeof("useless")];
  memcpy(av0, "useless", sizeof("useless"));

  // Set av1.
  char* av1 = NULL;
  std::string str("--smp=");
  str += std::to_string(_core_number);
  av1 = new char[str.size() + 1]();
  memcpy(av1, str.c_str(), str.size());

  // Set av2.
  char* av2 = NULL;
  if (DisablePinCores(_job_name)) {
    std::string thread_affinity("--thread-affinity=0");
    av2 = new char[thread_affinity.size() + 1]();
    memcpy(av2, thread_affinity.c_str(), thread_affinity.size());
  } else {
    GetCpuset(&av2);
  }

  // Set av3 if necessary.
  char* av3 = NULL;
  if (EnablePolling(_job_name)) {
    ++(*argc);
    std::string poll_mode("--poll-mode");
    av3 = new char[poll_mode.size() + 1]();
    memcpy(av3, poll_mode.c_str(), poll_mode.size());
  }

  // Allocate one extra char for 'NULL' at the end.
  *argv = new char*[(*argc) + 1]();
  (*argv)[0] = av0;
  (*argv)[1] = av1;
  (*argv)[2] = av2;
  if (av3 != NULL) {
    (*argv)[3] = av3;
  }

  LOG(INFO) << "Construct args result, argc: " << *(argc)
            << ", argv[0]: " << (*argv)[0]
            << ", argv[1]: " << (*argv)[1]
            << ", argv[2]: " << (*argv)[2];
  if (av3 != NULL) {
    LOG(INFO) << "argv[3]:" << (*argv)[3];
  }
}

void SeastarEngine::AsyncStart() {
  int argc = 0;
  char** argv = NULL;

  ConstructArgs(&argc, &argv);
  seastar::app_template app;
  app.run_deprecated(argc, argv, [&] {
    //todo should elegant stop distribute server & client
    //seastar::engine().at_exit([&] { return _server.stop(); });
    //seastar::engine().at_exit([&] { return _client.stop(); });

    LOG(INFO) << "Begin to start seastar engine.";
    return _server.start().then([this] {
      LOG(INFO) << "Seastar engine starting...";
      return _server.invoke_on_all(&SeastarServer::start, _local, _tag_factory);
    }).then([this]() {
        _init_ready.store(true, std::memory_order_relaxed);
        LOG(INFO) << "Seastar engine started succefully, listen port: " << _local << ".";
        return seastar::make_ready_future();
      }); 
  });
}

} // namespace tensorflow

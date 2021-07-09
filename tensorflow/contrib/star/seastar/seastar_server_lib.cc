#include <fstream>
#include <map>
#include <stdexcept>

#include "grpc/support/alloc.h"
#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"
#include "tensorflow/contrib/star/seastar/seastar_channel_cache.h"
#include "tensorflow/contrib/star/seastar/seastar_engine.h"
#include "tensorflow/contrib/star/seastar/seastar_server_lib.h"
#include "tensorflow/contrib/star/seastar/seastar_worker_cache.h"
#include "tensorflow/contrib/star/star_channel_spec.h"
#include "tensorflow/contrib/star/star_worker_service.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_resource.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"


namespace tensorflow {

SeastarServer::SeastarServer(const ServerDef& server_def, Env* env)
  : StarServerBase(server_def, env) {}

SeastarServer::~SeastarServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());

  delete seastar_engine_;
}

Status SeastarServer::StarWorkerCacheFactory(
    const WorkerCacheFactoryOptions& options,
    WorkerCacheInterface** worker_cache) {
  if (options.job_name == nullptr || options.job_name->empty()) {
    Status s = errors::InvalidArgument(
        "The master (current machine) is not included in the provided "
        "cluster_def. ",
        options.cluster_def->DebugString());
    LOG(WARNING) << s;
    return s;
  }

  StarChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(options, &channel_spec));
  std::unique_ptr<SeastarChannelCache> channel_cache(
      NewSeastarChannelCache(seastar_engine_, channel_spec));

  string name_prefix = strings::StrCat("/job:", *options.job_name, "/replica:0",
                                       "/task:", options.task_index);

  const string host_port = channel_cache->TranslateTask(name_prefix);
  int requested_port;

  if (!strings::safe_strto32(str_util::Split(host_port, ':')[1],
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                           channel_cache->TranslateTask(name_prefix), "\".");
  }

  LOG(INFO) << "SeastarWorkerCacheFactory, name_prefix:" << name_prefix;
  *worker_cache = NewSeastarWorkerCacheWithLocalWorker(
      channel_cache.release(), worker_impl_, name_prefix, &worker_env_);

  if (master_env_.run_graph_mode || master_env_.run_graph_mode_v2) {
    for (auto device : master_env_.local_devices) {
      ResourceMgr *rm = device->resource_manager();
      WorkerResource *worker_resource = new WorkerResource();
      worker_resource->worker_cache = *worker_cache;
      rm->Create("worker_resource", "worker_resource", worker_resource);
    }
  }

  return Status::OK();
}

void SeastarServer::CreateEngine(size_t server_number, const string& job_name) {
  seastar_engine_ = new SeastarEngine(server_number, star_bound_port_,
                                      job_name, worker_service_);
}

Status SeastarServer::Create(const ServerDef& server_def, Env* env,
                             std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<StarServerBase> ret(
      new SeastarServer(server_def, env == nullptr ? Env::Default() : env));

  TF_RETURN_IF_ERROR(ret->Init());
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {
class SeastarServerFactory : public ServerFactory {
public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return (server_def.protocol() == "grpc++" ||
            ((server_def.protocol() == "star_server" ||
              server_def.protocol() == "star_server_v2") &&
             server_def.job_name() != "ps"));
  }

  Status NewServer(const ServerDef& server_def,
      std::unique_ptr<ServerInterface>* out_server) override {
    return SeastarServer::Create(server_def, Env::Default(), out_server);
  }
};

class SeastarServerRegistrar {
public:
  SeastarServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    memset(&alloc_fns, 0, sizeof(alloc_fns));
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("SEASTAR_SERVER", new SeastarServerFactory());
}
};

static SeastarServerRegistrar registrar;

} // namespace

} // namespace tensorflow

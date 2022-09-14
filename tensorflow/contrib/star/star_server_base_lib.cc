#include <fstream>
#include <map>
#include <stdexcept>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"
#include "grpc/support/alloc.h"
#include "tensorflow/contrib/star/star_channel_spec.h"
#include "tensorflow/contrib/star/star_rendezvous_mgr.h"
#include "tensorflow/contrib/star/star_server_base_lib.h"
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
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"


namespace tensorflow {
namespace {
const char* kEndpointMapFile = ".endpoint_map";
} // namespace

class StarPortMgr {
public:
  explicit StarPortMgr(
      const ServerDef& server_def) {
    local_star_port_ = -1;
    ParseGrpcServerDef(server_def);
    LoadEndpointMapForFile();
  }

  std::string GetStarIpPort(const std::string& job_name, int task_index) {
    const auto it_job = grpc_cluster_spec_.find(job_name);
    if (it_job == grpc_cluster_spec_.end()) {
      LOG(FATAL) << "[Distributed] Job name: " << job_name
                 << " does not exist in cluster spec.";
    }
    const std::map<int, std::string>& task_map = it_job->second;

    const auto it_task = task_map.find(task_index);
    if (it_task == task_map.end()) {
      LOG(FATAL) << "[Distributed] Job name: "
                 << job_name << ", task index: " << task_index
                 << " does not exist in cluster spec.";
    }
    const std::string& grpc_ip_port = it_task->second;

    const auto it_star = endpoint_grpc2star_.find(grpc_ip_port);
    if (it_star == endpoint_grpc2star_.end()) {
      LOG(FATAL) << "[Distributed] Star ip and port "
                 << "not found for job name: " << job_name
                 << "task index: " << task_index << ".";
    }

    return it_star->second;
  }

  int GetLocalStarPort() {
    const auto it = endpoint_grpc2star_.find(local_grpc_ip_port_);
    if (it == endpoint_grpc2star_.end()) {
      LOG(FATAL) << "[Distributed] Star ip and port "
                 << "not found for job name: " << job_name_
                 << "task index: " << task_index_ << ".";
    }
    const std::string& local_star_ip_port = it->second;
    int local_star_port = -1;

    const auto& vec = str_util::Split(local_star_ip_port, ":");
    CHECK_EQ(vec.size(), 2);

    strings::safe_strto32(vec[1], &local_star_port);
    CHECK_GT(local_star_port, 0);

    return local_star_port;
  }

  std::string get_job_name() const {
    return job_name_;
  }

private:
  void ParseGrpcServerDef(const ServerDef& server_def) {
    job_name_ = server_def.job_name();
    task_index_ = server_def.task_index();

    for (const auto& job : server_def.cluster().job()) {
      auto& task_map = grpc_cluster_spec_[job.name()];
      for (const auto& task : job.tasks()) {
        task_map[task.first] = task.second;
        if (job.name() == job_name_ && task.first == task_index_) {
          local_grpc_ip_port_ = task.second;
        }
      }
    }

    if (local_grpc_ip_port_.empty()) {
      LOG(FATAL) << "[Distributed] Job name: " << job_name_
                 << ", task index: " << task_index_
                 << " not found in cluter spec.";
    }
  }

  void LoadEndpointMapForFile() {
    string endpointmap_path;
    ReadStringFromEnvVar("TF_SEASTAR_ENDPOINT_MAP_PATH", "", &endpointmap_path);
    string endpointmap_file = io::JoinPath(endpointmap_path, kEndpointMapFile);
    std::ifstream fin(endpointmap_file, std::ios::in);
    if (!fin.good()) {
      LOG(FATAL) << "[Distributed] Load endpoint map from "
                 << endpointmap_file << " failed.";
    }

    string str;
    while (getline(fin, str)) {
      std::vector<std::string> vec = str_util::Split(str, '=');
      CHECK_EQ(vec.size(), 2);
      endpoint_grpc2star_[vec[0]] = vec[1];
    }
  }

private:
  typedef std::map<std::string, std::string> EndpointMap;
  typedef std::map<std::string, std::map<int, std::string>>
    GrpcClusterSpec;

  EndpointMap endpoint_grpc2star_;
  GrpcClusterSpec  grpc_cluster_spec_;
  std::string job_name_;
  int task_index_;
  std::string local_grpc_ip_port_;
  int local_star_port_;
};

StarServerBase::StarServerBase(const ServerDef& server_def, Env* env)
  : server_def_(server_def), env_(env), state_(NEW) {
  star_port_mgr_ = new StarPortMgr(server_def_);
}

StarServerBase::~StarServerBase() {
  delete worker_impl_;
  delete master_service_;
  delete worker_service_;

  if (worker_env_.session_mgr != nullptr) {
    delete worker_env_.session_mgr;
  } else {
    delete worker_env_.device_mgr;
  }

  delete star_port_mgr_;
}

Status StarServerBase::Init() {
  mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);
  master_env_.env = env_;
  worker_env_.env = env_;

  SessionOptions sess_opts;
  ConfigProto config = server_def_.default_session_config();
  sess_opts.config = config;
  master_env_.run_graph_mode = GetRunGraphModeFlag(config);
  master_env_.run_graph_mode_lite = GetRunGraphModeFlagLite(config);
  if (master_env_.run_graph_mode ||
      master_env_.run_graph_mode_lite) {
    master_env_.run_graph_mode_with_zero_copy = true;
  }

  string name_prefix =
    strings::StrCat("/job:", server_def_.job_name(), "/replica:0",
        "/task:", server_def_.task_index());

  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(
      DeviceFactory::AddDevices(sess_opts, name_prefix, &devices));
  worker_env_.device_mgr = new DeviceMgr(std::move(devices));
  master_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.rendezvous_mgr = CreateRendezvousMgr(&worker_env_);

  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
        &default_worker_name, &unused)) {
    return errors::Internal(
        "[Distributed] Could not parse worker name.");
  }

  int requested_port = -1;
  for (const auto& job : server_def_.cluster().job()) {
    if (job.name() == server_def_.job_name()) {
      auto iter = job.tasks().find(server_def_.task_index());
      if (iter == job.tasks().end()) {
        return errors::InvalidArgument(
            "[Distributed] Task ", server_def_.task_index(),
            " was not defined in job \"",
            server_def_.job_name(), "\"");
      }

      const auto& hostname_port = str_util::Split(iter->second, ':');

      if (hostname_port.size() != 2) {
        return errors::InvalidArgument(
            "[Distributed] Could not parse port for local server from \"",
            iter->second, "\"");
      }

      if (!strings::safe_strto32(hostname_port[1], &requested_port)) {
        return errors::InvalidArgument(
            "[Distributed] Could not parse port for local server from \"",
            iter->second, "\"");
      }
      break;
    }
  }

  if (requested_port == -1) {
    return errors::Internal("[Distributed] Job \"",
        server_def_.job_name(), "\" was not defined in cluster");
  }

  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port),
      GetServerCredentials(server_def_), &bound_port_);
  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());

  master_impl_ = CreateMaster(&master_env_);
  master_service_ = NewGrpcMasterService(
      master_impl_.get(), config, &builder);

  server_ = builder.BuildAndStart();
  if (!server_) {
    return errors::Unknown(
        "[Distributed] Could not start gRPC server");
  }

  LOG(INFO) << "[Distributed] Starting grpc server, bind port:"
            << bound_port_;

  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  worker_impl_ = CreateWorker(&worker_env_);
  worker_service_ = CreateWorkerService(worker_impl_);

  worker_env_.compute_pool = ComputePool(sess_opts);
  star_bound_port_ = star_port_mgr_->GetLocalStarPort();
  size_t server_number = ParseServers(worker_cache_factory_options);

  CreateEngine(server_number, star_port_mgr_->get_job_name());

  WorkerCacheInterface* worker_cache;
  TF_RETURN_IF_ERROR(
      StarWorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);

  worker_env_.session_mgr = CreateSessionMgr(
      &worker_env_, SessionMgr::WorkerNameFromServerDef(server_def_),
      worker_cache,
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return StarWorkerCacheFactory(options, worker_cache);
      });

  // master intialize
  master_env_.tensor_fuse = config.tensor_fuse();
  master_env_.compute_pool = worker_env_.compute_pool;
  master_env_.ops = OpRegistry::Global();
  master_env_.worker_cache = worker_cache;
  master_env_.master_session_factory =
    [config](
        SessionOptions options, const MasterEnv* env,
        std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
        std::unique_ptr<WorkerCacheInterface> worker_cache,
        std::unique_ptr<DeviceSet> device_set,
	std::vector<string> filtered_worker_list) {
      options.config.MergeFrom(config);
      return new MasterSession(options, env, std::move(remote_devs),
          std::move(worker_cache), std::move(device_set),
	  std::move(filtered_worker_list),
          CreateNoOpStatsPublisher);
    };

  master_env_.worker_cache_factory =
    [this](const WorkerCacheFactoryOptions& options,
        WorkerCacheInterface** worker_cache) {
      return StarWorkerCacheFactory(options, worker_cache);
    };
  LocalMaster::Register(target(), master_impl_.get(),
      config.operation_timeout_in_ms());

  return Status::OK();
}

Status StarServerBase::ParseChannelSpec(
    const WorkerCacheFactoryOptions& options,
    StarChannelSpec* channel_spec) {
  for (const auto& job : options.cluster_def->job()) {
    std::map<int, string> host_ports;
    for (const auto& task : job.tasks()) {
      string& host_port = host_ports[task.first];
      if (!host_port.empty()) {
        return errors::InvalidArgument(
            "[Distributed] JobDef for job \"",
            job.name(), "\" specified two addresses for task \"",
            task.first, "\": ", host_port, " and ", task.second);
      }
      if (job.name() == *options.job_name &&
          task.first == options.task_index) {
        host_port = strings::StrCat("localhost:", star_bound_port_);
      } else {
        host_port = task.second;
        std::string star_host_port;
        int grpc_port = -1;
        const auto& vec = str_util::Split(host_port, ':');
        if (vec.size() != 2 ||
            !strings::safe_strto32(vec[1], &grpc_port)) {
          LOG(ERROR) << "[Distributed] Error host port schema "
                     << host_port;
          return errors::Cancelled(
              "[Distributed] error host port schema ",
              host_port);
        }

        star_host_port = star_port_mgr_->GetStarIpPort(
            job.name(), task.first);
        LOG(INFO) << "[Distributed] host port: "
                  << host_port << ", remote star host port: "
                  << star_host_port;
        host_port = star_host_port;
      }
    }

    TF_RETURN_IF_ERROR(
        channel_spec->AddHostPortsJob(job.name(), host_ports));
  }

  return Status::OK();
}

size_t StarServerBase::ParseServers(
    const WorkerCacheFactoryOptions& options) {
  size_t hosts_count = 0;
  for (const auto& job : options.cluster_def->job()) {
    hosts_count += job.tasks().size();
  }
  return hosts_count;
}

Status StarServerBase::Start() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW: {
      master_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_master_service",
                            [this] { master_service_->HandleRPCsLoop(); }));
      state_ = STARTED;
      LOG(INFO) << "[Distributed] Started server with target: "
                << target();
      return Status::OK();
    }
    case STARTED:
      LOG(INFO) << "[Distributed] Server already started (target: "
                << target() << ")";
      return Status::OK();
    case STOPPED:
      return errors::FailedPrecondition(
          "[Distributed] Server has stopped.");
    default:
      LOG(FATAL);
  }
  return Status::OK();
}

Status StarServerBase::Stop() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
      LOG(WARNING) <<
        "[Distributed] Clean shutdown is not currently implemented";
      return errors::Unimplemented(
          "[Distributed] Clean shutdown is not currently implemented");
    case STOPPED:
      LOG(INFO) << "[Distributed] Server already stopped (target: "
                << target() << ")";
      return Status::OK();
    default:
      LOG(FATAL);
  }
  return Status::OK();
}

Status StarServerBase::Join() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      // Prevent the server from being started subsequently.
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
    case STOPPED:
      master_thread_.reset();
      return Status::OK();
    default:
      LOG(FATAL);
  }
  return Status::OK();
}

const string StarServerBase::target() const {
  return strings::StrCat("grpc://localhost:", bound_port_);
}

std::shared_ptr<::grpc::ServerCredentials> StarServerBase::GetServerCredentials(
    const ServerDef& server_def) const {
  return ::grpc::InsecureServerCredentials();
}

std::unique_ptr<Master> StarServerBase::CreateMaster(MasterEnv* master_env) {
  return std::unique_ptr<Master>(new Master(master_env, 0.0));
}

RendezvousMgrInterface* StarServerBase::CreateRendezvousMgr(WorkerEnv* env) {
  return new StarRendezvousMgr(env);
}

StarWorker* StarServerBase::CreateWorker(WorkerEnv* env) {
  return NewStarWorker(&worker_env_).release();
}

SessionMgr* StarServerBase::CreateSessionMgr(WorkerEnv* env,
    const string& default_worker_name,
    WorkerCacheInterface* default_worker_cache,
        std::function<Status(const ServerDef&,
            WorkerCacheInterface**)> worker_cache_factory) {
  return new SessionMgr(env, default_worker_name,
      std::unique_ptr<WorkerCacheInterface>(default_worker_cache),
      worker_cache_factory);
}

StarWorkerService* StarServerBase::CreateWorkerService(
    StarWorker* worker) {
  return NewStarWorkerService(worker).release();
}

bool StarServerBase::GetRunGraphModeFlag(const ConfigProto& config) {
  // NOTE(jiankeng.pt): run_graph_mode is only
  // accepted in star_server protocol.
  // We do NOT want to add the run_graph_mode flag in
  // config.proto that will make the api_compatibility_test failed.
  // return config.run_graph_mode();
  return server_def_.protocol() == "star_server";
}

bool StarServerBase::GetRunGraphModeFlagLite(const ConfigProto& config) {
  return server_def_.protocol() == "star_server_lite";
}

} // namespace tensorflow

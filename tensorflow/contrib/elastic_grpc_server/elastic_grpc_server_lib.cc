/* Copyright 2023 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#include "tensorflow/contrib/elastic_grpc_server/elastic_grpc_server_lib.h"

#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "include/json/json.h"
#include "grpc/support/alloc.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/util/env_var.h"

#include "tensorflow/contrib/elastic_grpc_server/elastic_service.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_resource.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/protobuf/cluster.pb.h"

namespace tensorflow {

namespace {

// static utility function
RendezvousMgrInterface* NewRpcRendezvousMgr(const WorkerEnv* env) {
  return new RpcRendezvousMgr(env);
}

}  // namespace

ElasticGrpcServer::ElasticGrpcServer(const ServerDef& server_def, Env* env)
    : GrpcServer(server_def, env) {}

ElasticGrpcServer::~ElasticGrpcServer() {
  delete elastic_service_;
}

Status ElasticGrpcServer::UpdateServerDef(const string& cluster_def_str, int& before_part_num, int& after_part_num) {
  std::string tf_config;
  ReadStringFromEnvVar("TF_CONFIG", "", &tf_config);
  if (!tf_config.empty()) {
    Json::Reader reader;
    Json::Value tf_config_json;
    if(!reader.parse(tf_config, tf_config_json)) {
      return errors::Internal("PARSE TF_CONFIG ERROR");
    }
    if ((tf_config_json["cluster"].isNull()) ||
        (tf_config_json["cluster"]["ps"].isNull())) {
      return errors::Internal("PARSE PS FROM TF_CONFIG ERROR");
    }

    Json::Value cluster_json;
    if (!reader.parse(cluster_def_str, cluster_json)) {
      LOG(ERROR) << "cluster_def is not correct with " << cluster_def_str;
      return errors::Internal("PARSE TF_CONFIG/cluster ERROR");
    }

    std::unordered_set<string> ps_addrs_vec;
    after_part_num = cluster_json["cluster"]["ps"].size();
    for (auto& value: cluster_json["cluster"]["ps"]) {
      ps_addrs_vec.emplace(value.asString());
    }

    int job_size = server_def_.cluster().job_size();
    for (int j = 0; j < job_size; ++j) {
      auto* job = server_def_.mutable_cluster()->mutable_job(j);
      if (job->name() == "ps") {
        before_part_num = job->tasks_size();
        if (before_part_num == after_part_num) {
          return Status::OK();
        } else if (after_part_num > before_part_num) {
          int idx = before_part_num;
          LOG(INFO) << "SCALING UP, partition_num is: " << after_part_num;
          std::unordered_set<string> target_string_set;
          for (auto& value: tf_config_json["cluster"]["ps"]) {
            target_string_set.emplace(value.asString());
          }
          for (auto ps_addr: ps_addrs_vec) {
            if (target_string_set.find(ps_addr) == target_string_set.end()) {
              job->mutable_tasks()->insert({idx, ps_addr});
              tf_config_json["cluster"]["ps"].append(ps_addr);
            }
          } 
          break;
        } else {
          LOG(INFO) << "SCALING DOWN, partition_num is: " << after_part_num;
          for (int i = 0; i < before_part_num; ++i) {
            string tmp_string = tf_config_json["cluster"]["ps"][i].asString();
            if (ps_addrs_vec.find(tmp_string) == ps_addrs_vec.end()) {
              Json::Value ps_addr;
              tf_config_json["cluster"]["ps"].removeIndex(i, &ps_addr);
              job->mutable_tasks()->erase(i);
            }
          }
        }
      }
    }
    Json::FastWriter writer;
    std::string new_tf_config = writer.write(tf_config_json);
    LOG(INFO) << "new TF_CONFIG " << new_tf_config;
    setenv("TF_CONFIG", new_tf_config.c_str(), 1);
  }
  return Status::OK();
}

Status ElasticGrpcServer::Update(const string& cluster_def_str) {
  int before_part_num, after_part_num;
  Status s = UpdateServerDef(cluster_def_str, before_part_num, after_part_num);
  if (!s.ok()) {
    LOG(ERROR) << s.error_message();
    return Status::OK();
  }

  if (after_part_num == before_part_num) {
    return Status::OK();
  }

  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);
  ConfigProto config = server_def_.default_session_config();
  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env()->local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }
  std::unique_ptr<DeviceResolverDistributed> dev_resolver(
      new DeviceResolverDistributed(worker_env()->device_mgr, worker_cache,
                                    default_worker_name));
  std::unique_ptr<CollectiveParamResolverDistributed> param_resolver(
      new CollectiveParamResolverDistributed(config, worker_env()->device_mgr,
                                              dev_resolver.get(), worker_cache,
                                              default_worker_name));
  worker_env()->collective_executor_mgr = new RpcCollectiveExecutorMgr(
      config, worker_env()->device_mgr, std::move(dev_resolver),
      std::move(param_resolver), worker_cache, default_worker_name);

  if (worker_env()->session_mgr != nullptr) {
    delete worker_env()->session_mgr;  // Deletes graph_mgr's.
  }

  // Set up worker environment.
  worker_env()->session_mgr = new SessionMgr(
      worker_env(), SessionMgr::WorkerNameFromServerDef(server_def_),
      std::unique_ptr<WorkerCacheInterface>(worker_cache),
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return WorkerCacheFactory(options, worker_cache);
      });
  master_env()->worker_cache = worker_cache;
  // Finish setting up master environment.
  
  StatsPublisherFactory stats_factory = opts_.stats_factory;
  master_env()->master_session_factory =
      [config, stats_factory](
          SessionOptions options, const MasterEnv* env,
          std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
          std::unique_ptr<WorkerCacheInterface> worker_cache,
          std::unique_ptr<DeviceSet> device_set,
          std::vector<string> filtered_worker_list) {
        options.config.MergeFrom(config);
        return new MasterSession(options, env, std::move(remote_devs),
                                 std::move(worker_cache), std::move(device_set),
                                 std::move(filtered_worker_list),
                                 stats_factory);
      };
  master_env()->worker_cache_factory =
      [this](const WorkerCacheFactoryOptions& options,
             WorkerCacheInterface** worker_cache) {
        return WorkerCacheFactory(options, worker_cache);
      };
  return Status::OK();
}

void ElasticGrpcServer::MaybeMutateBuilder(::grpc::ServerBuilder* builder) {
  elastic_service_ = NewElasticGrpcService(this, builder);
}

Status ElasticGrpcServer::Start() {
  {
    mutex_lock l(mu_);
    switch (state_) {
      case NEW: {
        update_server_thread_.reset(
            env_->StartThread(ThreadOptions(), "TF_elastic_service",
                              [this] { elastic_service_->HandleRPCsLoop(); }));
        LOG(INFO) << "Started server with target: " << target();
        break;
      }
      case STARTED:
        LOG(INFO) << "Server already started (target: " << target() << ")";
        return Status::OK();
      case STOPPED:
        return errors::FailedPrecondition("Server has stopped.");
      default:
        LOG(FATAL);
    }
  }
  return GrpcServer::Start();
}

Status ElasticGrpcServer::Join() {
  GrpcServer::Join();
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      LOG(FATAL) << "Server shoud already closed";
    case STARTED:
    case STOPPED:
      update_server_thread_.reset();  
      return Status::OK();
    default:
      LOG(FATAL);
  }
}

/* static */
Status ElasticGrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<ElasticGrpcServer> ret(
      new ElasticGrpcServer(server_def, env == nullptr ? Env::Default() : env));
  ServiceInitFunction service_func = nullptr;
  GrpcServerOptions options;
  options.rendezvous_mgr_func = NewRpcRendezvousMgr;
  Status s = ret->Init(options);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
  *out_server = std::move(ret);
  return Status::OK();
}

/* static */
Status ElasticGrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ElasticGrpcServer>* out_server) {
  std::unique_ptr<ElasticGrpcServer> ret(
      new ElasticGrpcServer(server_def, env == nullptr ? Env::Default() : env));
  GrpcServerOptions options;
  options.rendezvous_mgr_func = NewRpcRendezvousMgr;
  Status s = ret->Init(options);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class ElasticGrpcServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "elastic-grpc";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return ElasticGrpcServer::Create(server_def, Env::Default(), out_server);
  }
};

// Registers a `ServerFactory` for `ElasticGrpcServer` instances.
class ElasticGrpcServerRegistrar {
 public:
  ElasticGrpcServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    memset(&alloc_fns, 0, sizeof(alloc_fns));
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("ELASTIC_GRPC_SERVER", new ElasticGrpcServerFactory());
  }
};
static ElasticGrpcServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow
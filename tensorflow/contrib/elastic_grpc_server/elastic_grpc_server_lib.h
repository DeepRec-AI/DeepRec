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

#ifndef TENSORFLOW_CONTRIB_ELASTIC_GRPC_SERVER_ELASTIC_GRPC_SERVER_LIB_H_
#define TENSORFLOW_CONTRIB_ELASTIC_GRPC_SERVER_ELASTIC_GRPC_SERVER_LIB_H_

#include <memory>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/stats_publisher_interface.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class ElasticGrpcServer : public GrpcServer {
 public:
  ElasticGrpcServer(const ServerDef& server_def, Env* env);
  
  virtual ~ElasticGrpcServer() override;

  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ElasticGrpcServer>* out_server);

  Status Update(const string& cluster_def_str);

  void MaybeMutateBuilder(::grpc::ServerBuilder* builder) override;

  Status Start() override;
  
  Status Join() override;

 private:
  Status UpdateServerDef(const string& cluster_def_str, int& before_part_num, int& after_part_num);
  
 private:
  // TensorFlow Eager implementation, and RPC polling thread.
  AsyncServiceInterface* elastic_service_ = nullptr;
  std::unique_ptr<Thread> update_server_thread_ GUARDED_BY(mu_);

  std::unique_ptr<::grpc::Server> server_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_ELASTIC_GRPC_SERVER_ELASTIC_GRPC_SERVER_LIB_H_
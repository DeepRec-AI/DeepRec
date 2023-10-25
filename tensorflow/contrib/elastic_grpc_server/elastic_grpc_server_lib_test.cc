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
#include "tensorflow/core/lib/core/status_test_util.h"

#include "gtest/gtest.h"

namespace tensorflow {

class ElasticGrpcServerTest : public ::testing::Test {
 protected:
  Status FillServerDef(const string& job_spec, ServerDef* options) {
    options->set_protocol("elastic-grpc");
    options->set_job_name("chief");
    options->set_task_index(0);

    uint32 my_tasks_per_replica = 0;
    for (const string& job_str : str_util::Split(job_spec, ',')) {
        JobDef* job_def = options->mutable_cluster()->add_job();
        // Split each entry in the flag into 2 pieces, separated by "|".
        const std::vector<string> job_pieces = str_util::Split(job_str, '|');
        CHECK_EQ(2, job_pieces.size()) << job_str;
        job_def->set_name(job_pieces[0]);
        // Does a bit more validation of the tasks_per_replica.
        const StringPiece spec = job_pieces[1];
        // job_str is of form <job_name>|<host_ports>.
        const std::vector<string> host_ports = str_util::Split(spec, ';');
        uint32 tasks_per_replica = host_ports.size();
        for (size_t i = 0; i < host_ports.size(); ++i) {
        (*job_def->mutable_tasks())[i] = host_ports[i];
        }
        if (job_def->name() == options->job_name()) {
        my_tasks_per_replica = tasks_per_replica;
        }
        LOG(INFO) << "Peer " << job_def->name() << " " << tasks_per_replica << " {"
                << absl::StrJoin(host_ports, ", ") << "}";
    }
    if (my_tasks_per_replica == 0) {
        return errors::InvalidArgument("Invalid job specification");
    }
    return Status::OK();
  }  
};

//Test Update Logic
TEST_F(ElasticGrpcServerTest, UpdateServer) {
  Status s;
  std::unique_ptr<ElasticGrpcServer> grpc_server;
  ServerDef server_def;
  std::string job_spec = "worker|localhost:2222,ps|localhost:10086;localhost:10087;localhost:10088,chief|localhost:2220";
  TF_ASSERT_OK(FillServerDef(job_spec, &server_def));
  s = ElasticGrpcServer::Create(server_def, Env::Default(), &grpc_server);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create server: " << s.error_message();
  }
  TF_ASSERT_OK(grpc_server->Start());
  // TF_QCHECK_OK(grpc_server->Join());
  LOG(INFO) << "SCALING DOWN";
  std::string tf_config_str = "{\"cluster\": {\"worker\": [\"localhost:2222\"],\"ps\": [\"localhost:10086\", \"localhost:10087\"],\"chief\": [\"localhost:2220\"]]}}";
  grpc_server->Update(tf_config_str);
  LOG(INFO) << "SCALING UP";
  tf_config_str = "{\"cluster\": {\"worker\": [\"localhost:2222\"],\"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\"],\"chief\": [\"localhost:2220\"]]}}";
  grpc_server->Update(tf_config_str);
  grpc_server.release();
}

}
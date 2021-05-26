#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_SERVER_LIB_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_SERVER_LIB_H_

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "tensorflow/contrib/star/star_server_base_lib.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class SeastarEngine;

class SeastarServer : public StarServerBase {
 public:
  SeastarServer(const ServerDef& server_def, Env* env);
  virtual ~SeastarServer();
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);

 protected:
  virtual Status StarWorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                                        WorkerCacheInterface** worker_cache);
  virtual void CreateEngine(size_t server_number, const string& job_name);

 private:
  SeastarEngine* seastar_engine_ = nullptr;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_SERVER_LIB_H_

#ifndef ODL_PROCESSOR_MODEL_STORE_SPARSE_STORE_FACTORY_H_
#define ODL_PROCESSOR_MODEL_STORE_SPARSE_STORE_FACTORY_H_
#include <string>

#include "odl_processor/model_store/sparse_storage_interface.h"
#include "odl_processor/model_store/redis_sparse_store.h"

namespace tensorflow {
namespace processor {

class ModelStoreFactory {
 public:
  ~ModelStoreFactory() {}
  static AbstractModelStore* CreateModelStore(const std::string& type) {
    if ("local_redis" == type) {
      LocalRedis::Config config;
      config.ip = "127.0.0.1";
      config.port = 6379;
      static LocalRedis r(config);
      return &r;
    } else {
      LOG(WARNING) << "Not match any type";
    }
  }
 private:
  ModelStoreFactory() {}
};

} // namespace processor
} // namespace tensorflow
#endif // ODL_PROCESSOR_MODEL_STORE_SPARSE_STORE_FACTORY_H_

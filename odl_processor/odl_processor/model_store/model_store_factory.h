#ifndef ODL_PROCESSOR_MODEL_STORE_MODEL_STORE_FACTORY_H_
#define ODL_PROCESSOR_MODEL_STORE_MODEL_STORE_FACTORY_H_
#include <string>

#include "odl_processor/model_store/abs_model_store.h"
#include "odl_processor/model_store/redis_model_store.h"

namespace tensorflow {
namespace processor {

class ModelStoreFactory {
 public:
  ~ModelStoreFactory() {}
  static AbstractModelStore* CreateModelStore(const std::string& type) {
    if ("local_redis" == type) {
      typename LocalRedis::Config config;
      config.ip = "127.0.0.1";
      config.port = 6379;
      return new LocalRedis(config);
    } else {
      LOG(WARNING) << "Not match any type";
    }
  }
 private:
  ModelStoreFactory() {}
};

} // namespace processor
} // namespace tensorflow
#endif // ODL_PROCESSOR_MODEL_STORE_MODEL_STORE_FACTORY_H_

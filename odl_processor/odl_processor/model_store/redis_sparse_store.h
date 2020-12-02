#ifndef ODL_PROCESSOR_MODEL_STORE_REDIS_SPARSE_STORE_H_
#define ODL_PROCESSOR_MODEL_STORE_REDIS_SPARSE_STORE_H_
#include <vector>
#include <string>
#include <thread>

#include "odl_processor/model_store/sparse_storage_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

class redisAsyncContext;
struct event_base;

namespace tensorflow {
namespace processor {

class LocalRedis : public AbstractModelStore {
  public:
    struct Config {
      std::string ip;
      int32_t port = 0;
    };

    LocalRedis(Config config);

    ~LocalRedis();

    Status RegisterFeatures(const std::vector<std::string>& features) {
      return errors::Unimplemented("[Redis] Unimplement RegisterFeatures().");
    }

    Status Cleanup();

    Status BatchGet(uint64_t feature2id,
                    const Tensor& key,
                    const Tensor& value) {
      return errors::Unimplemented("[Redis] Unimplement BatchGet() in sync mode.");
    }


    Status BatchSet(uint64_t feature2id,
                    const Tensor& key,
                    const Tensor& value) {
      return errors::Unimplemented("[Redis] Unimplement BatchSet() in sync mode.");
    }


    Status BatchGetAsync(uint64_t feature2id,
                         const char* const keys,
                         char* const values,
                         size_t bytes_per_key,
                         size_t bytes_per_values,
                         size_t N,
                         const char* default_value,
                         BatchGetCallback cb);

    Status BatchSetAsync(uint64_t feature2id,
                         const char* const keys,
                         const char* const values,
                         size_t bytes_per_key,
                         size_t bytes_per_values,
                         size_t N,
                         BatchSetCallback cb);

  private:
    std::string ip_;
    int32_t port_;
    redisAsyncContext *ac_;
    event_base *base_;
    std::unique_ptr<std::thread> event_thread_;
};

class ClusterRedis : public AbstractModelStore {
  public:
    ~ClusterRedis() {}

    Status RegisterFeatures(const std::vector<std::string>& features) {
      return errors::Unimplemented("[Redis] Unimplement RegisterFeatures().");
    }

    Status Cleanup() {
      return errors::Unimplemented("[Redis] Unimplement Cleanup().");
    }

    Status BatchGet(uint64_t feature2id,
                    const Tensor& key,
                    const Tensor& value) {
      return errors::Unimplemented("[Redis] Unimplement BatchGet() in sync mode.");
    }


    Status BatchSet(uint64_t feature2id,
                    const Tensor& key,
                    const Tensor& value) {
      return errors::Unimplemented("[Redis] Unimplement BatchSet() in sync mode.");
    }


    Status BatchGetAsync(uint64_t feature2id,
                         const char* const keys,
                         char* const values,
                         size_t bytes_per_key,
                         size_t bytes_per_values,
                         size_t N,
                         const char* default_value,
                         BatchGetCallback cb) {
      return errors::Unimplemented("[Redis] Unimplement BatchGet() in async mode.");
    }

    Status BatchSetAsync(uint64_t feature2id,
                         const char* const keys,
                         const char* const values,
                         size_t bytes_per_key,
                         size_t bytes_per_values,
                         size_t N,
                         BatchSetCallback cb) {
      return errors::Unimplemented("[Redis] Unimplement BatchSet() in async mode.");
    }

};

} // namespace processor
} // namespace tensorflow

#endif // ODL_PROCESSOR_MODEL_STORE_REDIS_SPARSE_STORE_H_

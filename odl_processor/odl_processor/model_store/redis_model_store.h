#ifndef ODL_PROCESSOR_MODEL_STORE_REDIS_MODEL_STORE_H_
#define ODL_PROCESSOR_MODEL_STORE_REDIS_MODEL_STORE_H_
#include <vector>
#include <string>
#include <thread>

#include "odl_processor/model_store/abs_model_store.h"
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

    Status RegisterFeatures(const std::vector<std::string>& features);

    Status BatchGet(const std::string& feature,
                    const std::string& version,
                    const std::vector<char*>& keys,
                    size_t keys_byte_lens,
                    const std::vector<char*>& values) {
      return errors::Unimplemented("[Redis] Unimplement BatchGet() in sync mode.");
    }


    Status BatchSet(const std::string& feature,
                    const std::string& version,
                    const std::vector<char*>& keys,
                    size_t keys_byte_lens,
                    const std::vector<char*>& values,
                    size_t values_byte_lens) {
      return errors::Unimplemented("[Redis] Unimplement BatchSet() in sync mode.");
    }


    Status BatchGetAsync(const std::string& feature,
                         const std::string& version,
                         const std::vector<char*>& keys,
                         size_t keys_byte_lens,
                         const std::vector<char*>& values,
                         BatchGetCallback cb);

    Status BatchSetAsync(const std::string& feature,
                         const std::string& version,
                         const std::vector<char*>& keys,
                         size_t keys_byte_lens,
                         const std::vector<char*>& values,
                         size_t values_byte_lens,
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
      return Status::OK();
    }

    Status BatchGet(const std::string& feature,
                    const std::string& version,
                    const std::vector<char*>& keys,
                    size_t keys_byte_lens,
                    const std::vector<char*>& values) {
      return errors::Unimplemented("[Redis] Unimplement BatchGet() in sync mode.");
    }


    Status BatchSet(const std::string& feature,
                    const std::string& version,
                    const std::vector<char*>& keys,
                    size_t keys_byte_lens,
                    const std::vector<char*>& values,
                    size_t values_byte_lens) {
      return errors::Unimplemented("[Redis] Unimplement BatchSet() in sync mode.");
    }


    Status BatchGetAsync(const std::string& feature,
                         const std::string& version,
                         const std::vector<char*>& keys,
                         size_t keys_byte_lens,
                         const std::vector<char*>& values,
                         BatchGetCallback cb) {
      return errors::Unimplemented("[Redis] Unimplement BatchGet() in async mode.");
    }

    Status BatchSetAsync(const std::string& feature,
                         const std::string& version,
                         const std::vector<char*>& keys,
                         size_t keys_byte_lens,
                         const std::vector<char*>& values,
                         size_t values_byte_lens,
                         BatchSetCallback cb) {
      return errors::Unimplemented("[Redis] Unimplement BatchSet() in async mode.");
    }

};

} // namespace processor
} // namespace tensorflow

#endif // ODL_PROCESSOR_MODEL_STORE_REDIS_MODEL_STORE_H_

#ifndef ODL_PROCESSOR_MODEL_STORE_ABS_MODEL_STORE_H_
#define ODL_PROCESSOR_MODEL_STORE_ABS_MODEL_STORE_H_
#include <vector>
#include <string>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {

//     AbstractModelStore
//            ^
//        ____|_____
//       /    |     \
//      /     |      \
//     /      |       \
//  Local   Cluster  RocksDB
//  Redis    Redis    ...

typedef std::function<void(const Status&)> BatchGetCallback;
typedef std::function<void(const Status&)> BatchSetCallback;

class AbstractModelStore {
  public:
    virtual ~AbstractModelStore() {}

    // Register Feature
    virtual Status RegisterFeatures(const std::vector<std::string>& features) = 0;
    // Read Store
    virtual Status BatchGet(const std::string& feature,
                            const std::string& version,
                            const std::vector<char*>& keys,
                            size_t keys_byte_lens,
                            const std::vector<char*>& values) = 0;
    // Write Store
    virtual Status BatchSet(const std::string& feature,
                            const std::string& version,
                            const std::vector<char*>& keys,
                            size_t keys_byte_lens,
                            const std::vector<char*>& values,
                            size_t values_byte_lens) = 0;
    // Read Store Async
    virtual Status BatchGetAsync(const std::string& feature,
                                 const std::string& version,
                                 const std::vector<char*>& keys,
                                 size_t keys_byte_lens,
                                 const std::vector<char*>& values,
                                 BatchGetCallback cb) = 0;
    // Write Store Async
    virtual Status BatchSetAsync(const std::string& feature,
                                 const std::string& version,
                                 const std::vector<char*>& keys,
                                 size_t keys_byte_lens,
                                 const std::vector<char*>& values,
                                 size_t values_byte_lens,
                                 BatchSetCallback cb) = 0;
};

} // namespace processor
} // namespace tensorflow

#endif // ODL_PROCESSOR_MODEL_STORE_ABS_MODEL_STORE_H_

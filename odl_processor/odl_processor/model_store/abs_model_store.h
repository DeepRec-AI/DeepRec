#ifndef ODL_PROCESSOR_MODEL_STORE_ABS_MODEL_STORE_H_
#define ODL_PROCESSOR_MODEL_STORE_ABS_MODEL_STORE_H_
#include <vector>
#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {

//     AbstractModelStore
//             ^
//        _____|_____
//       /     |     \
//      /      |      \
//     /       |       \
//  Local   Cluster  RocksDB
//  Redis    Redis    ...

typedef std::function<void(const Status&)> BatchGetCallback;
typedef std::function<void(const Status&)> BatchSetCallback;

class AbstractModelStore {
  public:
    virtual ~AbstractModelStore() {}

    // Register Feature
    virtual Status RegisterFeatures(const std::vector<std::string>& features) = 0;
    // Cleanup Store
    virtual Status Cleanup() = 0;
    // Read Store
    virtual Status BatchGet(uint64_t feature2id,
                            const Tensor& key,
                            const Tensor& value) = 0;
    // Write Store
    virtual Status BatchSet(uint64_t feature2id,
                            const Tensor& key,
                            const Tensor& value
                            ) = 0;
    // Read Store Async
    virtual Status BatchGetAsync(uint64_t feature2id,        // featureID encode uint64
                                 const char* const keys,     // key buffer -- IDs
                                 char* const values,         // val buffer -- embeddings
                                 size_t bytes_per_key,       // sizeof(TKey)
                                 size_t bytes_per_values,    // sizeof(TValue) * embedding*dim
                                 size_t N,                   // embedding vocabulary size
                                 const char* default_value,  // embedding default buffer if ID NotFound
                                 BatchGetCallback cb) = 0;
    // Write Store Async
    virtual Status BatchSetAsync(uint64_t feature2id,        // featureID encode uint64
                                 const char* const keys,     // key buffer -- IDs
                                 const char* const values,   // val buffer -- embeddings
                                 size_t bytes_per_key,       // sizeof(TKey)
                                 size_t bytes_per_values,    // sizeof(TValue) * embedding*dim
                                 size_t N,                   // embedding vocabulary size
                                 BatchSetCallback cb) = 0;
};

} // namespace processor
} // namespace tensorflow

#endif // ODL_PROCESSOR_MODEL_STORE_ABS_MODEL_STORE_H_

#ifndef ODL_PROCESSOR_STORAGE_FEATURE_STORE_H_
#define ODL_PROCESSOR_STORAGE_FEATURE_STORE_H_
#include <vector>
#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {

typedef std::function<void(const Status&)> BatchGetCallback;
typedef std::function<void(const Status&)> BatchSetCallback;

class FeatureStore {
  public:
    virtual ~FeatureStore() {}

    // Cleanup Store
    virtual Status Cleanup() = 0;
    // Read Store Sync
    virtual Status BatchGet(uint64_t feature2id,             // featureID encode uint64
                            const char* const keys,          // key buffer -- IDs
                            char* const values,              // val buffer -- embeddings
                            size_t bytes_per_key,            // sizeof(TKey)
                            size_t bytes_per_values,         // sizeof(TValue) * embedding*dim
                            size_t N,                        // embedding vocabulary size
                            const char* default_value) = 0;  // embedding default buffer if ID NotFound
    // Write Store Sync
    virtual Status BatchSet(uint64_t feature2id,             // featureID encode uint64
                            const char* const keys,          // key buffer -- IDs
                            const char* const values,        // val buffer -- embeddings
                            size_t bytes_per_key,            // sizeof(TKey)
                            size_t bytes_per_values,         // sizeof(TValue) * embedding*dim
                            size_t N) = 0;                   // embedding vocabulary size
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

#endif // ODL_PROCESSOR_STORAGE_FEATURE_STORE_H_

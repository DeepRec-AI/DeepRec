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

struct StorageOptions {
  bool is_init_storage_;

  // for redis, connection will select DB 0~16
  size_t serving_storage_db_index_;
  size_t backup_storage_db_index_;

  StorageOptions(size_t serving_db_idx, size_t backup_db_idx)
      : serving_storage_db_index_(serving_db_idx),
        backup_storage_db_index_(backup_db_idx) {}

  StorageOptions(size_t serving_db_idx, size_t backup_db_idx,
                 bool is_init_storage)
      : is_init_storage_(is_init_storage),
        serving_storage_db_index_(serving_db_idx),
        backup_storage_db_index_(backup_db_idx) {}
};

struct StorageMeta {
  // for redis, db 0 ~ N

  // latest version in the storage,
  // full_version or delta version
  std::vector<int64_t> model_version;
  // current full version in the storage
  std::vector<int64_t> curr_full_version;
  // current db active or not
  std::vector<bool> active;
};

class FeatureStore {
  public:
    virtual ~FeatureStore() {}

    // Get meta data of storage if needed
    virtual Status GetStorageMeta(StorageMeta* meta) {
      return Status::OK();
    }

    virtual Status SetActiveStatus(bool active) {
      return Status::OK();
    }

    virtual Status GetModelVersion(int64_t* full_version,
                                   int64_t* latest_version) {
      return Status::OK();
    }

    virtual Status SetModelVersion(int64_t full_version,
                                   int64_t latest_version) {
      return Status::OK();
    }

    virtual Status GetStorageLock(int value, int timeout,
                                  bool* success) {
      return Status::OK();
    }

    virtual Status ReleaseStorageLock(int value) {
      return Status::OK();
    }

    // Cleanup Store
    virtual Status Cleanup() = 0;
    // Read Store Sync
    virtual Status BatchGet(uint64_t model_version,          // model version
                            uint64_t feature2id,             // featureID encode uint64
                            const char* const keys,          // key buffer -- IDs
                            char* const values,              // val buffer -- embeddings
                            size_t bytes_per_key,            // sizeof(TKey)
                            size_t bytes_per_values,         // sizeof(TValue) * embedding*dim
                            size_t N,                        // embedding vocabulary size
                            const char* default_value) = 0;  // embedding default buffer if ID NotFound
    // Write Store Sync
    virtual Status BatchSet(uint64_t model_version,          // model version
                            uint64_t feature2id,             // featureID encode uint64
                            const char* const keys,          // key buffer -- IDs
                            const char* const values,        // val buffer -- embeddings
                            size_t bytes_per_key,            // sizeof(TKey)
                            size_t bytes_per_values,         // sizeof(TValue) * embedding*dim
                            size_t N) = 0;                   // embedding vocabulary size
    // Read Store Async
    virtual Status BatchGetAsync(uint64_t model_version,     // model version
                                 uint64_t feature2id,        // featureID encode uint64
                                 const char* const keys,     // key buffer -- IDs
                                 char* const values,         // val buffer -- embeddings
                                 size_t bytes_per_key,       // sizeof(TKey)
                                 size_t bytes_per_values,    // sizeof(TValue) * embedding*dim
                                 size_t N,                   // embedding vocabulary size
                                 const char* default_value,  // embedding default buffer if ID NotFound
                                 BatchGetCallback cb) = 0;
    // Write Store Async
    virtual Status BatchSetAsync(uint64_t model_version,     // model version
                                 uint64_t feature2id,        // featureID encode uint64
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

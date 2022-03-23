/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SERVING_ODL_PROCESSOR_FRAMEWORK_FILESYSTEM_OSS_FILE_SYSTEM_H_
#define SERVING_ODL_PROCESSOR_FRAMEWORK_FILESYSTEM_OSS_FILE_SYSTEM_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "aos_log.h"
#include "aos_status.h"
#include "aos_string.h"
#include "aos_util.h"
#include "oss_api.h"
#include "oss_auth.h"
#include "oss_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace io {

/// Aliyun oss implementation of a file system.
class OSSFileSystem : public FileSystem {
 public:
  OSSFileSystem();

  Status NewRandomAccessFile(
      const string& filename,
      std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& filename,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  Status FileExists(const string& fname) override;

  Status Stat(const string& fname, FileStatistics* stat) override;

  Status GetChildren(const string& dir, std::vector<string>* result) override;

  Status GetMatchingPaths(const string& pattern,
                          std::vector<string>* results) override;

  Status DeleteFile(const string& fname) override;

  Status CreateDir(const string& dirname) override;

  Status RecursivelyCreateDir(const string& dirname) override;

  Status DeleteDir(const string& dirname) override;

  Status GetFileSize(const string& fname, uint64* file_size) override;

  Status RenameFile(const string& src, const string& target) override;

  Status IsDirectory(const string& fname) override;

  Status DeleteRecursively(const string& dirname, int64* undeleted_files,
                           int64* undeleted_dirs) override;

 private:
  Status _CreateDirInternal(aos_pool_t* pool,
                            const oss_request_options_t* options,
                            const string& bucket, const string& dirname);

  Status _StatInternal(aos_pool_t* pool, const oss_request_options_t* options,
                       const string& bucket, const string& object,
                       FileStatistics* stat);

  Status _DeleteObjectInternal(const oss_request_options_t* options,
                               const string& bucket, const string& object);

  Status _RetrieveObjectMetadata(aos_pool_t* pool,
                                 const oss_request_options_t* options,
                                 const string& bucket, const string& object,
                                 FileStatistics* stat);

  aos_status_t* _RenameFileInternal(const oss_request_options_t* oss_options,
                                    aos_pool_t* pool,
                                    const aos_string_t& source_bucket,
                                    const aos_string_t& source_object,
                                    const aos_string_t& dest_bucket,
                                    const aos_string_t& dest_object);

  Status _ListObjects(aos_pool_t* pool, const oss_request_options_t* options,
                      const string& bucket, const string& key,
                      std::vector<string>* result, bool return_all = true,
                      bool return_full_path = false,
                      bool should_remove_suffix = true,
                      int max_ret_per_iterator = 1000);

  Status _InitOSSCredentials();

  Status _ParseOSSURIPath(const StringPiece fname, std::string& bucket,
                          std::string& object, std::string& host,
                          std::string& access_id, std::string& access_key);

  // The number of bytes to read ahead for buffering purposes
  //  in the RandomAccessFile implementation. Defaults to 5Mb.
  const size_t read_ahead_bytes_ = 5 * 1024 * 1024;

  // The number of bytes for each upload part. Defaults to 64MB
  const size_t upload_part_bytes_ = 64 * 1024 * 1024;

  // The max number of attempts to upload a file to OSS using the resumable
  // upload API.
  const int32 max_upload_attempts_ = 5;

  mutex mu_;

  TF_DISALLOW_COPY_AND_ASSIGN(OSSFileSystem);
};

}  // namespace io
}  // namespace tensorflow

#endif  // SERVING_ODL_PROCESSOR_FRAMEWORK_FILESYSTEM_OSS_FILE_SYSTEM_H_


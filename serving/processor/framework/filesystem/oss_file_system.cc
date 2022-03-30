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
#include "serving/processor/framework/filesystem/oss_file_system.h"

#include <pwd.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "aos_string.h"
#include "oss_define.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace io {
namespace {

constexpr char kOSSCredentialsDefaultFile[] = ".osscredentials";
constexpr char kOSSCredentialsFileEnvKey[] = "OSS_CREDENTIALS";
constexpr char kOSSCredentialsSection[] = "OSSCredentials";
constexpr char kOSSCredentialsHostKey[] = "host";
constexpr char kOSSCredentialsAccessIdKey[] = "accessid";
constexpr char kOSSCredentialsAccesskeyKey[] = "accesskey";
constexpr char kOSSAccessIdKey[] = "id";
constexpr char kOSSAccessKeyKey[] = "key";
constexpr char kOSSHostKey[] = "host";
constexpr char kDelim[] = "/";

void oss_initialize_with_throwable() {
  if (aos_http_io_initialize(NULL, 0) != AOSE_OK) {
    throw std::exception();
  }
}

Status oss_initialize() {
  static std::once_flag initFlag;
  try {
    std::call_once(initFlag, [] { oss_initialize_with_throwable(); });
  } catch (...) {
    LOG(FATAL) << "can not init OSS connection";
    return errors::Internal("can not init OSS connection");
  }

  return Status::OK();
}

void oss_error_message(aos_status_s* status, std::string* msg) {
  *msg = status->req_id;
  if (aos_status_is_ok(status)) {
    return;
  }

  msg->append(" ");
  msg->append(std::to_string(status->code));

  if (status->code == 404) {
    msg->append(" object not exists!");
    return;
  }

  if (status->error_msg) {
    msg->append(" ");
    msg->append(status->error_msg);
    return;
  }
}

class OSSConnection {
 public:
  OSSConnection(const std::string& endPoint, const std::string& accessKey,
                const std::string& accessKeySecret) {
    aos_pool_create(&_pool, NULL);
    _options = oss_request_options_create(_pool);
    _options->config = oss_config_create(_options->pool);
    aos_str_set(&_options->config->endpoint, endPoint.c_str());
    aos_str_set(&_options->config->access_key_id, accessKey.c_str());
    aos_str_set(&_options->config->access_key_secret, accessKeySecret.c_str());
    _options->config->is_cname = 0;
    _options->ctl = aos_http_controller_create(_options->pool, 0);
  }

  ~OSSConnection() {
    if (NULL != _pool) {
      aos_pool_destroy(_pool);
    }
  }

  oss_request_options_t* getRequestOptions() { return _options; }

  aos_pool_t* getPool() { return _pool; }

 private:
  aos_pool_t* _pool = NULL;
  oss_request_options_t* _options = NULL;
};

class OSSRandomAccessFile : public RandomAccessFile {
 public:
  OSSRandomAccessFile(const std::string& endPoint, const std::string& accessKey,
                      const std::string& accessKeySecret,
                      const std::string& bucket, const std::string& object,
                      size_t read_ahead_bytes, size_t file_length)
      : shost(endPoint),
        sak(accessKey),
        ssk(accessKeySecret),
        sbucket(bucket),
        sobject(object),
        total_file_length_(file_length) {
    read_ahead_bytes_ = std::min(read_ahead_bytes, file_length);
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    // offset is 0 based, so last offset should be
    // just before total_file_length_
    if (offset >= total_file_length_) {
      return errors::OutOfRange("EOF reached, ", offset,
                                " is read out of file length ",
                                total_file_length_);
    }

    if (offset + n > total_file_length_) {
      n = total_file_length_ - offset;
    }

    VLOG(1) << "read " << sobject << " from " << offset << " to " << offset + n;

    mutex_lock lock(mu_);
    const bool range_start_included = offset >= buffer_start_offset_;
    const bool range_end_included =
        offset + n <= buffer_start_offset_ + buffer_size_;
    if (range_start_included && range_end_included) {
      // The requested range can be filled from the buffer.
      const size_t offset_in_buffer =
          std::min<uint64>(offset - buffer_start_offset_, buffer_size_);
      const auto copy_size = std::min(n, buffer_size_ - offset_in_buffer);
      VLOG(1) << "read from buffer " << offset_in_buffer << " to "
              << offset_in_buffer + copy_size << " total " << buffer_size_;
      std::copy(buffer_.begin() + offset_in_buffer,
                buffer_.begin() + offset_in_buffer + copy_size, scratch);
      *result = StringPiece(scratch, copy_size);
    } else {
      // Update the buffer content based on the new requested range.
      const size_t desired_buffer_size =
          std::min(n + read_ahead_bytes_, total_file_length_);
      if (n > buffer_.capacity() ||
          desired_buffer_size > 2 * buffer_.capacity()) {
        // Re-allocate only if buffer capacity increased significantly.
        VLOG(1) << "reserve buffer to " << desired_buffer_size;
        buffer_.reserve(desired_buffer_size);
      }

      buffer_start_offset_ = offset;
      VLOG(1) << "load buffer" << buffer_start_offset_;
      TF_RETURN_IF_ERROR(LoadBufferFromOSS(desired_buffer_size));

      // Set the results.
      memcpy(scratch, buffer_.data(), std::min(buffer_size_, n));
      *result = StringPiece(scratch, std::min(buffer_size_, n));
    }

    if (result->size() < n) {
      // This is not an error per se. The RandomAccessFile interface expects
      // that Read returns OutOfRange if fewer bytes were read than requested.
      return errors::OutOfRange("EOF reached, ", result->size(),
                                " bytes were read out of ", n,
                                " bytes requested.");
    }
    return Status::OK();
  }

 private:
  /// A helper function to actually read the data from OSS. This function loads
  /// buffer_ from OSS based on its current capacity.
  Status LoadBufferFromOSS(size_t desired_buffer_size) const
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    size_t range_start = buffer_start_offset_;
    size_t range_end = buffer_start_offset_ + std::min(buffer_.capacity() - 1,
                                                       desired_buffer_size - 1);
    range_end = std::min(range_end, total_file_length_ - 1);

    OSSConnection conn(shost, sak, ssk);
    aos_pool_t* _pool = conn.getPool();
    oss_request_options_t* _options = conn.getRequestOptions();
    aos_string_t bucket_;
    aos_string_t object_;
    aos_table_t* headers_;
    aos_list_t tmp_buffer;
    aos_table_t* resp_headers;

    aos_list_init(&tmp_buffer);
    aos_str_set(&_options->config->endpoint, shost.c_str());
    aos_str_set(&_options->config->access_key_id, sak.c_str());
    aos_str_set(&_options->config->access_key_secret, ssk.c_str());
    _options->config->is_cname = 0;
    _options->ctl = aos_http_controller_create(_options->pool, 0);
    aos_str_set(&bucket_, sbucket.c_str());
    aos_str_set(&object_, sobject.c_str());
    headers_ = aos_table_make(_pool, 1);

    std::string range("bytes=");
    range.append(std::to_string(range_start))
        .append("-")
        .append(std::to_string(range_end));
    apr_table_set(headers_, "Range", range.c_str());
    VLOG(1) << "read from OSS with " << range.c_str();

    aos_status_t* s =
        oss_get_object_to_buffer(_options, &bucket_, &object_, headers_, NULL,
                                 &tmp_buffer, &resp_headers);
    if (aos_status_is_ok(s)) {
      aos_buf_t* content = NULL;
      int64_t size = 0;
      int64_t pos = 0;
      buffer_.clear();
      buffer_size_ = 0;

      // copy data to local buffer
      aos_list_for_each_entry(aos_buf_t, content, &tmp_buffer, node) {
        size = aos_buf_size(content);
        std::copy(content->pos, content->pos + size, buffer_.begin() + pos);
        pos += size;
      }
      buffer_size_ = pos;
      return Status::OK();
    } else {
      string msg;
      oss_error_message(s, &msg);
      VLOG(0) << "read " << sobject << " failed, errMsg: " << msg;
      return errors::Internal("read failed: ", sobject, " errMsg: ", msg);
    }
  }

  std::string shost;
  std::string sak;
  std::string ssk;
  std::string sbucket;
  std::string sobject;
  const size_t total_file_length_;
  size_t read_ahead_bytes_;

  mutable mutex mu_;
  mutable std::vector<char> buffer_ GUARDED_BY(mu_);
  // The original file offset of the first byte in the buffer.
  mutable size_t buffer_start_offset_ GUARDED_BY(mu_) = 0;
  mutable size_t buffer_size_ GUARDED_BY(mu_) = 0;
};

class OSSReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  OSSReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {}
  const void* data() override { return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override { return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

class OSSWritableFile : public WritableFile {
 public:
  OSSWritableFile(const std::string& endPoint, const std::string& accessKey,
                  const std::string& accessKeySecret, const std::string& bucket,
                  const std::string& object, size_t part_size)
      : shost(endPoint),
        sak(accessKey),
        ssk(accessKeySecret),
        sbucket(bucket),
        sobject(object),
        part_size_(part_size),
        is_closed_(false),
        part_number_(1) {
    InitAprPool();
  }

  ~OSSWritableFile() { ReleaseAprPool(); }

  Status Append(StringPiece data) override {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(_CheckClosed());
    InitAprPool();
    if (CurrentBufferLength() >= part_size_) {
      TF_RETURN_IF_ERROR(_FlushInternal());
    }

    aos_buf_t* tmp_buf = aos_create_buf(pool_, data.size() + 1);
    aos_buf_append_string(pool_, tmp_buf, data.data(), data.size());
    aos_list_add_tail(&tmp_buf->node, &buffer_);
    return Status::OK();
  }

  Status Close() override {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(_CheckClosed());
    InitAprPool();
    TF_RETURN_IF_ERROR(_FlushInternal());
    aos_table_t* complete_headers = NULL;
    aos_table_t* resp_headers = NULL;
    aos_status_t* status = NULL;
    oss_list_upload_part_params_t* params = NULL;
    aos_list_t complete_part_list;
    oss_list_part_content_t* part_content = NULL;
    oss_complete_part_content_t* complete_part_content = NULL;
    aos_string_t upload_id;
    aos_str_set(&upload_id, upload_id_.c_str());

    params = oss_create_list_upload_part_params(pool_);
    aos_list_init(&complete_part_list);
    status = oss_list_upload_part(options_, &bucket_, &object_, &upload_id,
                                  params, &resp_headers);

    if (!aos_status_is_ok(status)) {
      string msg;
      oss_error_message(status, &msg);
      VLOG(0) << "List multipart " << sobject << " failed, errMsg: " << msg;
      return errors::Internal("List multipart failed: ", sobject,
                              " errMsg: ", msg);
    }

    aos_list_for_each_entry(oss_list_part_content_t, part_content,
                            &params->part_list, node) {
      complete_part_content = oss_create_complete_part_content(pool_);
      aos_str_set(&complete_part_content->part_number,
                  part_content->part_number.data);
      aos_str_set(&complete_part_content->etag, part_content->etag.data);
      aos_list_add_tail(&complete_part_content->node, &complete_part_list);
    }

    status = oss_complete_multipart_upload(options_, &bucket_, &object_,
                                           &upload_id, &complete_part_list,
                                           complete_headers, &resp_headers);

    if (!aos_status_is_ok(status)) {
      string msg;
      oss_error_message(status, &msg);
      VLOG(0) << "Complete multipart " << sobject << " failed, errMsg: " << msg;
      return errors::Internal("Complete multipart failed: ", sobject,
                              " errMsg: ", msg);
    }

    is_closed_ = true;
    return Status::OK();
  }

  Status Flush() override {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(_CheckClosed());
    if (CurrentBufferLength() >= part_size_) {
      InitAprPool();
      TF_RETURN_IF_ERROR(_FlushInternal());
    }

    return Status::OK();
  }

  Status Sync() override { return Flush(); }

 private:
  void InitAprPool() {
    if (NULL == pool_) {
      aos_pool_create(&pool_, NULL);
      options_ = oss_request_options_create(pool_);
      options_->config = oss_config_create(options_->pool);
      aos_str_set(&options_->config->endpoint, shost.c_str());
      aos_str_set(&options_->config->access_key_id, sak.c_str());
      aos_str_set(&options_->config->access_key_secret, ssk.c_str());
      options_->config->is_cname = 0;
      options_->ctl = aos_http_controller_create(options_->pool, 0);

      aos_str_set(&bucket_, sbucket.c_str());
      aos_str_set(&object_, sobject.c_str());

      headers_ = aos_table_make(pool_, 1);
      aos_list_init(&buffer_);
    }
  }

  void ReleaseAprPool() {
    if (NULL != pool_) {
      aos_pool_destroy(pool_);
      pool_ = NULL;
    }
  }

  Status _InitMultiUpload() {
    if (upload_id_.empty()) {
      aos_string_t uploadId;
      aos_status_t* status = NULL;
      aos_table_t* resp_headers = NULL;

      InitAprPool();
      status = oss_init_multipart_upload(options_, &bucket_, &object_,
                                         &uploadId, headers_, &resp_headers);

      if (!aos_status_is_ok(status)) {
        string msg;
        oss_error_message(status, &msg);
        VLOG(0) << "Init multipart upload " << sobject
                << " failed, errMsg: " << msg;
        return errors::Unavailable("Init multipart upload failed: ", sobject,
                                   " errMsg: ", msg);
      }

      upload_id_ = uploadId.data;
    }

    return Status::OK();
  }

  Status _FlushInternal() {
    aos_table_t* resp_headers = NULL;
    aos_status_s* status = NULL;
    aos_string_t uploadId;
    if (CurrentBufferLength() > 0) {
      _InitMultiUpload();

      aos_str_set(&uploadId, upload_id_.c_str());
      status =
          oss_upload_part_from_buffer(options_, &bucket_, &object_, &uploadId,
                                      part_number_, &buffer_, &resp_headers);

      if (!aos_status_is_ok(status)) {
        string msg;
        oss_error_message(status, &msg);
        VLOG(0) << "Upload multipart " << sobject << " failed, errMsg: " << msg;
        return errors::Internal("Upload multipart failed: ", sobject,
                                " errMsg: ", msg);
      }

      VLOG(1) << " upload " << sobject << " with part" << part_number_
              << " succ";
      part_number_++;
      ReleaseAprPool();
      InitAprPool();
    }
    return Status::OK();
  }

  const size_t CurrentBufferLength() { return aos_buf_list_len(&buffer_); }

  Status _CheckClosed() {
    if (is_closed_) {
      return errors::Internal("Already closed.");
    }

    return Status::OK();
  }

  std::string shost;
  std::string sak;
  std::string ssk;
  std::string sbucket;
  std::string sobject;
  size_t part_size_;

  aos_pool_t* pool_ = NULL;
  oss_request_options_t* options_ = NULL;
  aos_string_t bucket_;
  aos_string_t object_;
  aos_table_t* headers_ = NULL;
  aos_list_t buffer_;
  std::string upload_id_;

  bool is_closed_;
  mutex mu_;
  int64_t part_number_;
};
}  // namespace

OSSFileSystem::OSSFileSystem() {}

// Splits a oss path to endpoint bucket object and token
// For example
// "oss://bucket-name\x01id=accessid\x02key=accesskey\x02host=endpoint/path/to/file.txt"
Status OSSFileSystem::_ParseOSSURIPath(const StringPiece fname,
                                       std::string& bucket, std::string& object,
                                       std::string& host,
                                       std::string& access_id,
                                       std::string& access_key) {
  StringPiece scheme, bucketp, remaining;
  io::ParseURI(fname, &scheme, &bucketp, &remaining);

  if (scheme != "oss") {
    return errors::InvalidArgument("OSS path does not start with 'oss://':",
                                   fname);
  }

  str_util::ConsumePrefix(&remaining, kDelim);
  object = string(remaining);

  std::string bucketDelim = "?";
  std::string accessDelim = "&";
  if (bucketp.find('\x01') != StringPiece::npos) {
    bucketDelim = "\x01";
    accessDelim = "\x02";
  }

  // contains id, key, host information
  size_t pos = bucketp.find(bucketDelim);
  bucket = string(bucketp.substr(0, pos));
  StringPiece access_info = bucketp.substr(pos + 1);
  std::vector<std::string> access_infos =
      str_util::Split(access_info, accessDelim);
  for (const auto& key_value : access_infos) {
    StringPiece data(key_value);
    size_t pos = data.find('=');
    if (pos == StringPiece::npos) {
      return errors::InvalidArgument("OSS path access info faied: ", fname,
                                     " info:", key_value);
    }
    StringPiece key = data.substr(0, pos);
    StringPiece value = data.substr(pos + 1);
    if (str_util::StartsWith(key, kOSSAccessIdKey)) {
      access_id = string(value);
    } else if (str_util::StartsWith(key, kOSSAccessKeyKey)) {
      access_key = string(value);
    } else if (str_util::StartsWith(key, kOSSHostKey)) {
      host = string(value);
    } else {
      return errors::InvalidArgument("OSS path access info faied: ", fname,
                                     " unkown info:", key_value);
    }
  }

  if (bucket.empty()) {
    return errors::InvalidArgument("OSS path does not contain a bucket name:",
                                   fname);
  }

  if (access_id.empty() || access_key.empty() || host.empty()) {
    return errors::InvalidArgument(
        "OSS path does not contain valid access info:", fname);
  }

  VLOG(1) << "bucket: " << bucket << ",access_id: " << access_id
          << ",access_key: " << access_key << ",host: " << host;

  return Status::OK();
}

Status OSSFileSystem::NewRandomAccessFile(
    const std::string& filename, std::unique_ptr<RandomAccessFile>* result) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(filename, bucket, object, host, access_id, access_key));
  FileStatistics stat;
  OSSConnection conn(host, access_id, access_key);
  TF_RETURN_IF_ERROR(_RetrieveObjectMetadata(
      conn.getPool(), conn.getRequestOptions(), bucket, object, &stat));
  result->reset(new OSSRandomAccessFile(host, access_id, access_key, bucket,
                                        object, read_ahead_bytes_,
                                        stat.length));
  return Status::OK();
}

Status OSSFileSystem::NewWritableFile(const std::string& fname,
                                      std::unique_ptr<WritableFile>* result) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(fname, bucket, object, host, access_id, access_key));

  result->reset(new OSSWritableFile(host, access_id, access_key, bucket, object,
                                    upload_part_bytes_));
  return Status::OK();
}

Status OSSFileSystem::NewAppendableFile(const std::string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  return errors::Unimplemented(
      "Does not support appendable file in OSSFileSystem");
}

Status OSSFileSystem::NewReadOnlyMemoryRegionFromFile(
    const std::string& filename,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  uint64 size;
  TF_RETURN_IF_ERROR(GetFileSize(filename, &size));
  std::unique_ptr<char[]> data(new char[size]);

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(filename, &file));

  StringPiece piece;
  TF_RETURN_IF_ERROR(file->Read(0, size, &piece, data.get()));

  result->reset(new OSSReadOnlyMemoryRegion(std::move(data), size));
  return Status::OK();
}

Status OSSFileSystem::FileExists(const std::string& fname) {
  FileStatistics stat;
  if (Stat(fname, &stat).ok()) {
    return Status::OK();
  } else {
    return errors::NotFound(fname, " does not exists");
  }
}

// For GetChildren , we should not return prefix
Status OSSFileSystem::_ListObjects(
    aos_pool_t* pool, const oss_request_options_t* options,
    const std::string& bucket, const std::string& key,
    std::vector<std::string>* result, bool return_all, bool return_full_path,
    bool should_remove_suffix, int max_ret_per_iterator) {
  aos_string_t bucket_;
  aos_status_t* s = NULL;
  oss_list_object_params_t* params = NULL;
  oss_list_object_content_t* content = NULL;
  const char* next_marker = "";

  aos_str_set(&bucket_, bucket.c_str());
  params = oss_create_list_object_params(pool);
  params->max_ret = max_ret_per_iterator;
  aos_str_set(&params->prefix, key.c_str());
  aos_str_set(&params->marker, next_marker);

  do {
    s = oss_list_object(options, &bucket_, params, NULL);
    if (!aos_status_is_ok(s)) {
      string msg;
      oss_error_message(s, &msg);
      VLOG(0) << "can not list object " << key << " errMsg: " << msg;
      return errors::NotFound("can not list object:", key, " errMsg: ", msg);
    }

    aos_list_for_each_entry(oss_list_object_content_t, content,
                            &params->object_list, node) {
      int path_length = content->key.len;
      if (should_remove_suffix && path_length > 0 &&
          content->key.data[content->key.len - 1] == '/') {
        path_length = content->key.len - 1;
      }
      if (return_full_path) {
        string child(content->key.data, 0, path_length);
        result->push_back(child);
      } else {
        int prefix_len = (key.length() > 0 && key.at(key.length() - 1) != '/')
                             ? key.length() + 1
                             : key.length();
        // remove prefix for GetChildren
        if (content->key.len > prefix_len) {
          string child(content->key.data + prefix_len, 0,
                       path_length - prefix_len);
          result->push_back(child);
        }
      }
    }

    next_marker = apr_psprintf(pool, "%.*s", params->next_marker.len,
                               params->next_marker.data);

    aos_str_set(&params->marker, next_marker);
    aos_list_init(&params->object_list);
    aos_list_init(&params->common_prefix_list);
  } while (params->truncated == AOS_TRUE && return_all);

  return Status::OK();
}

Status OSSFileSystem::_StatInternal(aos_pool_t* pool,
                                    const oss_request_options_t* options,
                                    const std::string& bucket,
                                    const std::string& object,
                                    FileStatistics* stat) {
  Status s = _RetrieveObjectMetadata(pool, options, bucket, object, stat);
  if (s.ok()) {
    VLOG(1) << "RetrieveObjectMetadata for object: " << object
            << " file success";
    return s;
  }

  // add suffix
  std::string objectName = object + kDelim;
  s = _RetrieveObjectMetadata(pool, options, bucket, objectName, stat);
  if (s.ok()) {
    VLOG(1) << "RetrieveObjectMetadata for object: " << objectName
            << " directory success";
    stat->is_directory = true;
    return s;
  }

  // check list if it has children
  std::vector<std::string> listing;
  s = _ListObjects(pool, options, bucket, object, &listing, true, false, false,
                   10);

  if (s == Status::OK() && !listing.empty()) {
    if (str_util::EndsWith(object, "/")) {
      stat->is_directory = true;
    }
    stat->length = 0;
    VLOG(1) << "RetrieveObjectMetadata for object: " << object
            << " get children success";
    return s;
  }

  VLOG(1) << "_StatInternal for object: " << object
          << ", failed with bucket: " << bucket;
  return errors::NotFound("can not find ", object);
}

Status OSSFileSystem::_RetrieveObjectMetadata(
    aos_pool_t* pool, const oss_request_options_t* options,
    const std::string& bucket, const std::string& object,
    FileStatistics* stat) {
  aos_string_t oss_bucket;
  aos_string_t oss_object;
  aos_table_t* headers = NULL;
  aos_table_t* resp_headers = NULL;
  aos_status_t* status = NULL;
  char* content_length_str = NULL;
  char* object_date_str = NULL;

  if (object.empty()) {  // root always exists
    stat->is_directory = true;
    stat->length = 0;
    return Status::OK();
  }

  aos_str_set(&oss_bucket, bucket.c_str());
  aos_str_set(&oss_object, object.c_str());
  headers = aos_table_make(pool, 0);

  status = oss_head_object(options, &oss_bucket, &oss_object, headers,
                           &resp_headers);
  if (aos_status_is_ok(status)) {
    content_length_str = (char*)apr_table_get(resp_headers, OSS_CONTENT_LENGTH);
    if (content_length_str != NULL) {
      stat->length = static_cast<int64>(atoll(content_length_str));
      VLOG(1) << "_RetrieveObjectMetadata object: " << object
              << " , with length: " << stat->length;
    }

    object_date_str = (char*)apr_table_get(resp_headers, "Last-Modified");
    if (object_date_str != NULL) {
      // the time is GMT Date, format like below
      // Date: Fri, 24 Feb 2012 07:32:52 GMT
      std::tm tm = {};
      strptime(object_date_str, "%a, %d %b %Y %H:%M:%S", &tm);
      stat->mtime_nsec = static_cast<int64>(mktime(&tm) * 1e9);

      VLOG(1) << "_RetrieveObjectMetadata object: " << object
              << " , with time: " << stat->mtime_nsec;
    } else {
      VLOG(0) << "find " << object << " with no datestr";
      return errors::NotFound("find ", object, " with no datestr");
    }

    if (object[object.length() - 1] == '/') {
      stat->is_directory = true;
    } else {
      stat->is_directory = false;
    }

    return Status::OK();
  } else {
    string msg;
    oss_error_message(status, &msg);
    VLOG(1) << "can not find object: " << object << ", with bucket: " << bucket
            << ", errMsg: " << msg;
    return errors::NotFound("can not find ", object, " errMsg: ", msg);
  }
}

Status OSSFileSystem::Stat(const std::string& fname, FileStatistics* stat) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(fname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* ossOptions = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();

  return _StatInternal(pool, ossOptions, bucket, object, stat);
}

Status OSSFileSystem::GetChildren(const std::string& dir,
                                  std::vector<std::string>* result) {
  result->clear();
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dir, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  return _ListObjects(pool, oss_options, bucket, object, result, true, false,
                      true, 1000);
}

Status OSSFileSystem::GetMatchingPaths(const std::string& pattern,
                                       std::vector<std::string>* results) {
  return tensorflow::internal::GetMatchingPaths(this, Env::Default(), pattern,
                                                results);
}

Status OSSFileSystem::_DeleteObjectInternal(
    const oss_request_options_t* options, const std::string& bucket,
    const std::string& object) {
  aos_string_t bucket_;
  aos_string_t object_;
  aos_table_t* resp_headers = NULL;
  aos_status_t* s = NULL;

  aos_str_set(&bucket_, bucket.c_str());
  aos_str_set(&object_, object.c_str());

  s = oss_delete_object(options, &bucket_, &object_, &resp_headers);
  if (!aos_status_is_ok(s)) {
    string msg;
    oss_error_message(s, &msg);
    VLOG(0) << "delete " << object << " failed, errMsg: " << msg;
    return errors::Internal("delete failed: ", object, " errMsg: ", msg);
  }

  return Status::OK();
}

Status OSSFileSystem::DeleteFile(const std::string& fname) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(fname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();

  return _DeleteObjectInternal(oss_options, bucket, object);
}

Status OSSFileSystem::CreateDir(const std::string& dirname) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dirname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* ossOptions = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  StringPiece dirs(object);

  std::vector<std::string> splitPaths =
      str_util::Split(dirs, '/', str_util::SkipEmpty());
  if (splitPaths.size() < 2) {
    return _CreateDirInternal(pool, ossOptions, bucket, object);
  }

  FileStatistics stat;
  StringPiece parent = io::Dirname(dirs);

  if (!_StatInternal(pool, ossOptions, bucket, string(parent), &stat).ok()) {
    VLOG(0) << "CreateDir() failed with bucket: " << bucket
            << ", parent: " << parent;
    return errors::Internal("parent does not exists: ", parent);
  }

  if (!stat.is_directory) {
    return errors::Internal("can not mkdir because parent is a file: ", parent);
  }

  TF_RETURN_IF_ERROR(_CreateDirInternal(pool, ossOptions, bucket, object));
  return Status::OK();
}

Status OSSFileSystem::RecursivelyCreateDir(const string& dirname) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dirname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* ossOptions = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  StringPiece dirs(object);

  std::vector<std::string> splitPaths =
      str_util::Split(dirs, '/', str_util::SkipEmpty());
  if (splitPaths.size() < 2) {
    return _CreateDirInternal(pool, ossOptions, bucket, object);
  }

  std::string dir = "";
  for (auto path : splitPaths) {
    dir.append(path + kDelim);

    if (!_CreateDirInternal(pool, ossOptions, bucket, dir).ok()) {
      VLOG(0) << "create dir failed with bucket: " << bucket
              << ", dir: " << dir;
      return errors::Internal("create dir failed: ", dir);
    }
  }

  return Status::OK();
}

Status OSSFileSystem::_CreateDirInternal(aos_pool_t* pool,
                                         const oss_request_options_t* options,
                                         const std::string& bucket,
                                         const std::string& dirname) {
  FileStatistics stat;
  if (_RetrieveObjectMetadata(pool, options, bucket, dirname, &stat).ok()) {
    if (!stat.is_directory) {
      VLOG(0) << "object already exists as a file: " << dirname;
      return errors::AlreadyExists("object already exists as a file: ",
                                   dirname);
    } else {
      return Status::OK();
    }
  }
  std::string object = dirname;
  if (dirname.at(dirname.length() - 1) != '/') {
    object += '/';
  }

  aos_status_t* s;
  aos_table_t* headers;
  aos_table_t* resp_headers;
  aos_string_t bucket_;
  aos_string_t object_;
  const char* data = "";
  aos_list_t buffer;
  aos_buf_t* content;

  aos_str_set(&bucket_, bucket.c_str());
  aos_str_set(&object_, object.c_str());
  headers = aos_table_make(pool, 0);

  aos_list_init(&buffer);
  content = aos_buf_pack(options->pool, data, strlen(data));
  aos_list_add_tail(&content->node, &buffer);
  s = oss_put_object_from_buffer(options, &bucket_, &object_, &buffer, headers,
                                 &resp_headers);

  if (aos_status_is_ok(s)) {
    return Status::OK();
  } else {
    string msg;
    oss_error_message(s, &msg);
    VLOG(1) << "mkdir " << dirname << " failed, errMsg: " << msg;
    return errors::Internal("mkdir failed: ", dirname, " errMsg: ", msg);
  }
}

Status OSSFileSystem::DeleteDir(const std::string& dirname) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dirname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  std::vector<std::string> children;
  Status s;

  s = _ListObjects(pool, oss_options, bucket, object, &children, true, false,
                   false, 10);
  if (s.ok() && !children.empty()) {
    return errors::FailedPrecondition("Cannot delete a non-empty directory.");
  }

  s = _DeleteObjectInternal(oss_options, bucket, object);

  if (s.ok()) {
    return s;
  }

  // Maybe should add slash
  return _DeleteObjectInternal(oss_options, bucket, object.append(kDelim));
}

Status OSSFileSystem::GetFileSize(const std::string& fname, uint64* file_size) {
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(fname, &stat));
  *file_size = stat.length;
  return Status::OK();
}

Status OSSFileSystem::RenameFile(const std::string& src,
                                 const std::string& target) {
  TF_RETURN_IF_ERROR(oss_initialize());
  std::string sobject, sbucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(src, sbucket, sobject, host, access_id, access_key));
  std::string dobject, dbucket;
  std::string dhost, daccess_id, daccess_key;
  TF_RETURN_IF_ERROR(_ParseOSSURIPath(target, dbucket, dobject, dhost,
                                      daccess_id, daccess_key));

  if (host != dhost || access_id != daccess_id || access_key != daccess_key) {
    VLOG(0) << "rename " << src << " to " << target << " failed, with errMsg: "
            << " source oss cluster does not match dest oss cluster";
    return errors::Internal(
        "rename ", src, " to ", target, " failed, errMsg: ",
        "source oss cluster does not match dest oss cluster");
  }

  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();

  aos_status_t* resp_status;
  aos_string_t source_bucket;
  aos_string_t source_object;
  aos_string_t dest_bucket;
  aos_string_t dest_object;

  aos_str_set(&source_bucket, sbucket.c_str());
  aos_str_set(&dest_bucket, dbucket.c_str());

  Status status = IsDirectory(src);
  if (status.ok()) {
    if (!str_util::EndsWith(sobject, "/")) {
      sobject += "/";
    }
    if (!str_util::EndsWith(dobject, "/")) {
      dobject += "/";
    }
    std::vector<std::string> childPaths;
    _ListObjects(pool, oss_options, sbucket, sobject, &childPaths, true, false,
                 false, 1000);
    for (const auto& child : childPaths) {
      std::string tmp_sobject = sobject + child;
      std::string tmp_dobject = dobject + child;

      aos_str_set(&source_object, tmp_sobject.c_str());
      aos_str_set(&dest_object, tmp_dobject.c_str());

      resp_status =
          _RenameFileInternal(oss_options, pool, source_bucket, source_object,
                              dest_bucket, dest_object);
      if (!aos_status_is_ok(resp_status)) {
        string msg;
        oss_error_message(resp_status, &msg);
        VLOG(0) << "rename " << src << " to " << target
                << " failed, with specific file:  " << tmp_sobject
                << ", with errMsg: " << msg;
        return errors::Internal("rename ", src, " to ", target,
                                " failed, errMsg: ", msg);
      }
      _DeleteObjectInternal(oss_options, sbucket, tmp_sobject);
    }
  }

  aos_str_set(&source_object, sobject.c_str());
  aos_str_set(&dest_object, dobject.c_str());
  resp_status = _RenameFileInternal(oss_options, pool, source_bucket,
                                    source_object, dest_bucket, dest_object);
  if (!aos_status_is_ok(resp_status)) {
    string msg;
    oss_error_message(resp_status, &msg);
    VLOG(0) << "rename " << src << " to " << target
            << " failed, errMsg: " << msg;
    return errors::Internal("rename ", src, " to ", target,
                            " failed, errMsg: ", msg);
  }

  return _DeleteObjectInternal(oss_options, sbucket, sobject);
}

aos_status_t* OSSFileSystem::_RenameFileInternal(
    const oss_request_options_t* oss_options, aos_pool_t* pool,
    const aos_string_t& source_bucket, const aos_string_t& source_object,
    const aos_string_t& dest_bucket, const aos_string_t& dest_object) {
  aos_status_t* resp_status;
  aos_table_t* resp_headers;
  aos_table_t* headers = aos_table_make(pool, 0);
  aos_string_t upload_id;

  oss_list_upload_part_params_t* list_upload_part_params;
  oss_upload_part_copy_params_t* upload_part_copy_params =
      oss_create_upload_part_copy_params(pool);
  oss_list_part_content_t* part_content;
  aos_list_t complete_part_list;
  oss_complete_part_content_t* complete_content;
  aos_table_t* list_part_resp_headers = NULL;
  aos_table_t* complete_resp_headers = NULL;
  int max_ret = 1000;

  // get file size
  FileStatistics stat;
  _StatInternal(pool, oss_options, std::string(source_bucket.data),
                std::string(source_object.data), &stat);
  uint64 file_size = stat.length;

  // file size bigger than upload_part_bytes_, need to split into multi parts
  if (file_size > upload_part_bytes_) {
    resp_status =
        oss_init_multipart_upload(oss_options, &dest_bucket, &dest_object,
                                  &upload_id, headers, &resp_headers);
    if (aos_status_is_ok(resp_status)) {
      VLOG(1) << "init multipart upload succeeded, upload_id is %s"
              << upload_id.data;
    } else {
      return resp_status;
    }

    // process for each single part
    int parts = ceil(double(file_size) / double(upload_part_bytes_));
    for (int i = 0; i < parts - 1; i++) {
      int64_t range_start = i * upload_part_bytes_;
      int64_t range_end = (i + 1) * upload_part_bytes_ - 1;
      int part_num = i + 1;

      aos_str_set(&upload_part_copy_params->source_bucket, source_bucket.data);
      aos_str_set(&upload_part_copy_params->source_object, source_object.data);
      aos_str_set(&upload_part_copy_params->dest_bucket, dest_bucket.data);
      aos_str_set(&upload_part_copy_params->dest_object, dest_object.data);
      aos_str_set(&upload_part_copy_params->upload_id, upload_id.data);

      upload_part_copy_params->part_num = part_num;
      upload_part_copy_params->range_start = range_start;
      upload_part_copy_params->range_end = range_end;

      headers = aos_table_make(pool, 0);

      resp_status = oss_upload_part_copy(oss_options, upload_part_copy_params,
                                         headers, &resp_headers);
      if (aos_status_is_ok(resp_status)) {
        VLOG(1) << "upload part " << part_num << " copy succeeded";
      } else {
        return resp_status;
      }
    }

    int64_t range_start = (parts - 1) * upload_part_bytes_;
    int64_t range_end = file_size - 1;

    aos_str_set(&upload_part_copy_params->source_bucket, source_bucket.data);
    aos_str_set(&upload_part_copy_params->source_object, source_object.data);
    aos_str_set(&upload_part_copy_params->dest_bucket, dest_bucket.data);
    aos_str_set(&upload_part_copy_params->dest_object, dest_object.data);
    aos_str_set(&upload_part_copy_params->upload_id, upload_id.data);
    upload_part_copy_params->part_num = parts;
    upload_part_copy_params->range_start = range_start;
    upload_part_copy_params->range_end = range_end;

    headers = aos_table_make(pool, 0);

    resp_status = oss_upload_part_copy(oss_options, upload_part_copy_params,
                                       headers, &resp_headers);
    if (aos_status_is_ok(resp_status)) {
      VLOG(1) << "upload part " << parts << " copy succeeded";
    } else {
      return resp_status;
    }

    headers = aos_table_make(pool, 0);
    list_upload_part_params = oss_create_list_upload_part_params(pool);
    list_upload_part_params->max_ret = max_ret;
    aos_list_init(&complete_part_list);
    resp_status = oss_list_upload_part(oss_options, &dest_bucket, &dest_object,
                                       &upload_id, list_upload_part_params,
                                       &list_part_resp_headers);
    aos_list_for_each_entry(oss_list_part_content_t, part_content,
                            &list_upload_part_params->part_list, node) {
      complete_content = oss_create_complete_part_content(pool);
      aos_str_set(&complete_content->part_number,
                  part_content->part_number.data);
      aos_str_set(&complete_content->etag, part_content->etag.data);
      aos_list_add_tail(&complete_content->node, &complete_part_list);
    }

    resp_status = oss_complete_multipart_upload(
        oss_options, &dest_bucket, &dest_object, &upload_id,
        &complete_part_list, headers, &complete_resp_headers);
    if (aos_status_is_ok(resp_status)) {
      VLOG(1) << "complete multipart upload succeeded";
    }
  } else {
    resp_status =
        oss_copy_object(oss_options, &source_bucket, &source_object,
                        &dest_bucket, &dest_object, headers, &resp_headers);
  }

  return resp_status;
}

Status OSSFileSystem::IsDirectory(const std::string& fname) {
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(fname, &stat));

  return stat.is_directory
             ? Status::OK()
             : errors::FailedPrecondition(fname + " is not a directory");
}

Status OSSFileSystem::DeleteRecursively(const std::string& dirname,
                                        int64* undeleted_files,
                                        int64* undeleted_dirs) {
  if (!undeleted_files || !undeleted_dirs) {
    return errors::Internal(
        "'undeleted_files' and 'undeleted_dirs' cannot be nullptr.");
  }
  *undeleted_files = 0;
  *undeleted_dirs = 0;

  TF_RETURN_IF_ERROR(oss_initialize());
  std::string object, bucket;
  std::string host, access_id, access_key;
  TF_RETURN_IF_ERROR(
      _ParseOSSURIPath(dirname, bucket, object, host, access_id, access_key));
  OSSConnection oss(host, access_id, access_key);
  oss_request_options_t* oss_options = oss.getRequestOptions();
  aos_pool_t* pool = oss.getPool();
  std::vector<std::string> children;

  FileStatistics stat;
  Status s;
  s = _StatInternal(pool, oss_options, bucket, object, &stat);
  if (!s.ok() || !stat.is_directory) {
    *undeleted_dirs = 1;
    return errors::NotFound(dirname, " doesn't exist or not a directory.");
  }

  s = _ListObjects(pool, oss_options, bucket, object, &children, true, true,
                   false, 1000);
  if (!s.ok()) {
    // empty dir, just delete it
    return _DeleteObjectInternal(oss_options, bucket, object);
  }

  for (const auto& child : children) {
    s = _DeleteObjectInternal(oss_options, bucket, child);
    if (!s.ok()) {
      s = _StatInternal(pool, oss_options, bucket, child, &stat);
      if (s.ok()) {
        if (stat.is_directory) {
          ++*undeleted_dirs;
        } else {
          ++*undeleted_files;
        }
      }
    }
  }

  if (*undeleted_dirs == 0 && *undeleted_files == 0) {
    // delete directory itself.
    if (object.at(object.length() - 1) == '/') {
      return _DeleteObjectInternal(oss_options, bucket, object);
    } else {
      return _DeleteObjectInternal(oss_options, bucket, object.append(kDelim));
    }
  }
  return Status::OK();
}

namespace {

REGISTER_FILE_SYSTEM("oss", OSSFileSystem);

}  // namespace
}  // end namespace io
}  // end namespace tensorflow


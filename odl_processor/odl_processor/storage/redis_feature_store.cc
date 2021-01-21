#include <unistd.h>
#include <vector>
#include <string>

#include "odl_processor/storage/redis_feature_store.h"
#include "adapters/libevent.h"
#include "async.h"
#include "hiredis.h"
#include "tensorflow/core/lib/core/status.h"

#define DEBUG 0

namespace tensorflow {
namespace processor {
namespace {

struct SetCallbackWrapper {
  BatchSetCallback cb;
};

struct GetCallbackWrapper {
  BatchGetCallback cb;
  void* tensor_data;
  int stride_in_bytes;
  const void* default_value;
  ~GetCallbackWrapper() {
  }
};

struct CleanupCallbackWrapper {
  Status s;
  bool done;
};

void ConnectCallback(const redisAsyncContext *ac, int status) {
  if (status != REDIS_OK) {
    LOG(ERROR) << "Error: " << ac->errstr;
    return;
  }
  LOG(INFO) << "Connected...";
}

void DisconnectCallback(const redisAsyncContext *ac, int status) {
  if (status != REDIS_OK) {
    LOG(ERROR) << "Error: " << ac->errstr;
    return;
  }
  LOG(INFO) << "Disconnected...";
}

void GetCallback(redisAsyncContext *ac, void *r, void *privdata) {
  redisReply *reply = (redisReply*)r;
  GetCallbackWrapper* wrapper = (GetCallbackWrapper*)privdata;
  BatchGetCallback callback = std::move(wrapper->cb);
  if (reply == NULL) {
    if (ac->errstr) {
      Status s(error::Code::INTERNAL,
        "[Redis] run GetCallback failed: " + std::string(ac->errstr));
      callback(s);
    } else {
      Status s(error::Code::INTERNAL,
        "[Redis] run GetCallback failed: unknown error");
      callback(s);
    }
  } else {
    if (REDIS_REPLY_ARRAY == reply->type) {
      for (int i = 0; i < reply->elements; i++) {
        if (REDIS_REPLY_NIL == reply->element[i]->type) {
          memcpy(wrapper->tensor_data + i * wrapper->stride_in_bytes,
                 wrapper->default_value,
                 wrapper->stride_in_bytes);
        } else if (REDIS_REPLY_STRING == reply->element[i]->type) {
          memcpy(wrapper->tensor_data + i * wrapper->stride_in_bytes,
                 reply->element[i]->str,
                 reply->element[i]->len);
        } else {
          Status s(error::Code::INTERNAL,
            "[Redis] run GetCallback failed.");
          callback(s);
          delete wrapper;
          return;
        }
      }
      Status s;
      callback(s);
    } else {
      LOG(ERROR) << "Async Multi GET unexcept redis reply";
    }
  }
  delete wrapper;
  return;
}

void SetCallback(redisAsyncContext *ac, void *r, void *privdata) {
  redisReply *reply = (redisReply*)r;
  SetCallbackWrapper* wrapper = (SetCallbackWrapper*)privdata;
  BatchSetCallback callback = std::move(wrapper->cb);
  if (reply == NULL) {
    if (ac->errstr) {
      Status s(error::Code::INTERNAL,
        "[Redis] run SetCallback failed: " + std::string(ac->errstr));
      LOG(ERROR) << "errstr: " << ac->errstr;
      callback(s);
    } else {
      Status s(error::Code::INTERNAL,
        "[Redis] run SetCallback failed: unknown error");
      callback(s);
    }
  } else {
    Status s;
    callback(s);
  }
  delete wrapper;
  return;
}

void CleanupCallback(redisAsyncContext *ac, void *r, void *privdata) {
  redisReply *reply = (redisReply*)r;
  CleanupCallbackWrapper* wrapper = (CleanupCallbackWrapper*)privdata;
  if (reply == NULL) {
    if (ac->errstr) {
      Status s(error::Code::INTERNAL,
        "[Redis] run CleanupCallback failed: " + std::string(ac->errstr));
      LOG(ERROR) << "errstr: " << ac->errstr;
      wrapper->s = s;
    } else {
      Status s(error::Code::INTERNAL,
        "[Redis] run CleanupCallback failed: unknown error");
      wrapper->s = s;
    }
  } else {
    if (REDIS_REPLY_STATUS == reply->type) {
      Status s;
      wrapper->s = s;
    } else {
      LOG(ERROR) << "[Redis] run CleanupCallback failed: " << reply->str;
      Status s(error::Code::INTERNAL,
        "[Redis] run CleanupCallback failed: unknown error");
      wrapper->s = s;
    }
  }
  wrapper->done = true;
  return;
}
} // anonymous namespace

LocalRedis::LocalRedis(const Config& config)
  : ip_(config.ip),
    port_(config.port),
    db_idx_(config.db_idx),
    c_(nullptr) {
  assert((c_ = redisConnect(ip_.c_str(), port_)) != nullptr);

  // Authentication
  if (!config.passwd.empty()) {
    std::string auth_cmd = "AUTH " + config.passwd;
    redisReply *reply = (redisReply *)redisCommand(c_, auth_cmd.c_str());
    if (REDIS_REPLY_STATUS != reply->type) {
      LOG(FATAL) << "Redis authentication failed.";
    }
    freeReplyObject(reply);
  }

  // Connect to the specified db, default is db-0
  std::string select_db_cmd = "SELECT " + std::to_string(db_idx_);
  redisReply *reply = (redisReply *)redisCommand(c_, select_db_cmd.c_str());
  if (REDIS_REPLY_STATUS != reply->type) {
    LOG(FATAL) << "Redis select db failed. db idx is " << db_idx_;
  }
  freeReplyObject(reply);
}

LocalRedis::~LocalRedis() {
  LOG(INFO) << "~LocalRedis";
}

Status LocalRedis::Cleanup() {
  redisReply *r= (redisReply *)redisCommand(c_, "FLUSHDB");
  if (REDIS_REPLY_STATUS == r->type) {
    return Status::OK();
  } else {
    LOG(ERROR) << "redisCommand-FLUSHDB failed: " << r->str;
    return Status(error::Code::INTERNAL,
        "[Redis] run redisCommand-FLUSHDB failed. " + std::string(r->str));
  }
}

Status LocalRedis::BatchGet(uint64_t model_version,
                            uint64_t feature2id,
                            const char* const keys,
                            char* const values,
                            size_t bytes_per_key,
                            size_t bytes_per_values,
                            size_t N,
                            const char* default_value) {
  int64_t len = N;
  size_t keys_byte_lens = bytes_per_key;
  char ** argv = new char*[len + 1 ];
  size_t * argvlen = new size_t[len + 1 ];

  int j = 0;
  argv[j] = new char[5];
  memcpy(argv[j],"MGET",4);
  argvlen[j] = 4;
  ++j;

#if DEBUG
  for(int i = 0 ; i < len; i++) {
    std::string key = std::to_string(*(int64*)(keys + i*keys_byte_lens));
    std::string key2 = std::to_string(model_version) + "_" + \
                       std::to_string(feature2id) + "_" + key;
    size_t key_length = key2.size();
    argvlen[j] = key_length;
    argv[j] = new char[key_length];
    memcpy((void*)argv[j], key2.c_str(), key_length);
    j++;
  }
#else
  int size_feature2id = sizeof(feature2id);
  int size_model_version = sizeof(model_version);
  size_t key_length = keys_byte_lens + \
                      size_model_version + \
                      size_feature2id;
  for(int i = 0 ; i < len; i++) {
    argvlen[j] = key_length;
    argv[j] = new char[key_length];
    memcpy((void*)argv[j], &model_version, size_model_version);
    memcpy((void*)(argv[j] + size_model_version), &feature2id, sizeof(feature2id));
    memcpy((void*)(argv[j] + size_model_version + size_feature2id),
           keys + i * keys_byte_lens, keys_byte_lens);
    j++;
  }
#endif

  redisReply *reply = (redisReply *)redisCommandArgv(c_,
                          len + 1, const_cast<const char **>(argv), argvlen);

  for(int i = 0; i < j; i++) {
    delete [] argv[i];
    argv[i] = NULL;
  }
  delete []argv;
  delete []argvlen;

  if (REDIS_REPLY_ARRAY != reply->type) {
    Status s(error::Code::INTERNAL,
      "[Redis] run redisCommandArgv-MGET failed." + std::string(reply->str));
    return s;
  } else {
    for (int i = 0; i < reply->elements; i++) {
      if (REDIS_REPLY_NIL == reply->element[i]->type) {
        memcpy((void*)values + i * bytes_per_values,
               default_value,
               bytes_per_values);
      } else if (REDIS_REPLY_STRING == reply->element[i]->type) {
#if DEBUG
        std::string result(reply->element[i]->str);
        LOG(INFO) << "GET" << result;
        std::vector<std::string> strs = str_util::Split(result, "_");
        float v[strs.size()];
        for (int k =0; k < strs.size() -1; ++k) {
          v[k] = std::stof(strs[k]);
        }
        memcpy(values + i * bytes_per_values,
               v,
               (strs.size()-1) * sizeof(float));
#else
        memcpy(values + i * bytes_per_values,
               reply->element[i]->str,
               reply->element[i]->len);
#endif
      } else {
        Status s(error::Code::INTERNAL,
          "[Redis] run redisCommandArgv-MGET failed." + std::string(reply->str));
        return s;
      }
    }
  }
  return Status::OK();
}

Status LocalRedis::BatchSet(uint64_t model_version,
                            uint64_t feature2id,
                            const char* const keys,
                            const char* const values,
                            size_t bytes_per_key,
                            size_t bytes_per_values,
                            size_t N) {
  int64_t len = N;
  size_t keys_byte_lens = bytes_per_key;
  const size_t values_byte_lens = bytes_per_values;
  char ** argv = new char*[2*len + 1];
  size_t * argvlen = new size_t[2*len + 1];

  int j = 0;
  argv[j] = new char[5];
  memcpy(argv[j], "MSET", 4);
  argvlen[j] = 4;
  ++j;

#if DEBUG
  for(int i = 0 ; i < len; i++) {
    std::string key = std::to_string(*(int64*)(keys + i*keys_byte_lens));
    std::string key2 = std::to_string(model_version) + "_" + \
                       std::to_string(feature2id) + "_" + key;
    size_t key_length = key2.size();
    argvlen[j] = key_length;
    argv[j] = new char[key_length];
    memcpy((void*)argv[j], key2.c_str(), key_length);
    j++;

    std::string values2;
    for (int k = 0; k< bytes_per_values/sizeof(float); ++k) {
      values2 += std::to_string(*((float*)values + i * bytes_per_values/sizeof(float) + k )) + std::string("_");
    }
    LOG(INFO) << "SET" << values2;
    size_t values_byte_lens2 = values2.size();
    argvlen[j] = values_byte_lens2;
    argv[j] = new char[values_byte_lens2];
    memcpy((void*)argv[j], values2.c_str(), values_byte_lens2);
    j++;
  }
#else
  int size_model_version = sizeof(model_version);
  int size_feature2id = sizeof(feature2id);
  size_t key_length = keys_byte_lens + \
                      size_model_version + \
                      size_feature2id;
  for(int i = 0 ; i < len; i++) {
    argvlen[j] = key_length;
    argv[j] = new char[key_length];
    memcpy((void*)argv[j], &model_version, size_model_version);
    memcpy((void*)(argv[j] + size_model_version), &feature2id, size_feature2id);
    memcpy((void*)(argv[j] + size_model_version + size_feature2id),
           keys + i * keys_byte_lens, keys_byte_lens);
    j++;

    argvlen[j] = values_byte_lens;
    argv[j] = new char[values_byte_lens];
    memcpy((void*)argv[j], values + i*values_byte_lens, values_byte_lens);
    j++;
  }
#endif
  redisReply *reply = (redisReply *)redisCommandArgv(c_,
                          len * 2 + 1, const_cast<const char **>(argv), argvlen);

  for(int i = 0; i < j; i++) {
    delete [] argv[i];
    argv[i] = NULL;
  }
  delete []argv;
  delete []argvlen;

  if (REDIS_REPLY_STATUS != reply->type) {
    LOG(ERROR) << "redisCommandArgv-MSET failed: " << reply->str;
    return Status(error::Code::INTERNAL,
        "[Redis] run redisCommandArgv-MSET failed." + std::string(reply->str));
  }
  return Status::OK();
}

Status LocalRedis::BatchGetAsync(uint64_t model_version,
                                 uint64_t feature2id,
                                 const char* const keys,
                                 char* const values,
                                 size_t bytes_per_key,
                                 size_t bytes_per_values,
                                 size_t N,
                                 const char* default_value,
                                 BatchGetCallback cb) {
  return errors::Unimplemented("[redis] unimplement BatchGetAsync() in async mode.");
}

Status LocalRedis::BatchSetAsync(uint64_t model_version,
                                 uint64_t feature2id,
                                 const char* const keys,
                                 const char* const values,
                                 size_t bytes_per_key,
                                 size_t bytes_per_values,
                                 size_t N,
                                 BatchSetCallback cb) {
  return errors::Unimplemented("[redis] unimplement BatchSetAsync() in async mode.");
}

} // namespace processor
} // namespace tensorflow

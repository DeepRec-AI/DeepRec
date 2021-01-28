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

Status GetRedisMeta(redisContext *c, int db, StorageMeta *meta) {
  std::string select_db_cmd = "SELECT " + std::to_string(db);
  redisReply *reply = (redisReply *)redisCommand(c, select_db_cmd.c_str());
  if (REDIS_REPLY_STATUS != reply->type) {
    return Status(error::Code::INTERNAL,
        "[Redis] run redisCommand-SELECT failed. " +
        std::string(reply->str));
  }
  freeReplyObject(reply);

  std::string get_active_cmd = "GET active";
  reply = (redisReply *)redisCommand(c, get_active_cmd.c_str());
  if (REDIS_REPLY_NIL == reply->type) {
    meta->active.push_back(false);
  } else if (REDIS_REPLY_STRING == reply->type) {
    if (std::string(reply->str) == "0") {
      meta->active.push_back(false);
    } else {
      meta->active.push_back(true);
    }
  } else {
    freeReplyObject(reply);
    return Status(error::Code::INTERNAL,
        "[Redis] run redisCommand-GET active failed. " +
        std::string(reply->str));
  }
  freeReplyObject(reply);

  std::string get_model_version_cmd = "GET model_version";
  reply = (redisReply *)redisCommand(c, get_model_version_cmd.c_str());
  if (REDIS_REPLY_NIL == reply->type) {
    meta->model_version.push_back(-1);
    meta->curr_full_version.push_back(-1);
  } else if (REDIS_REPLY_STRING != reply->type) {
    freeReplyObject(reply);
    return Status(error::Code::INTERNAL,
        "[Redis] run redisCommand-GET model_version failed. " +
        std::string(reply->str));
  } else {
    std::string str(reply->str);
    if (str.find(",") == std::string::npos) {
      freeReplyObject(reply);
      return Status(error::Code::INTERNAL,
          "[Redis] Parse model_version failed. " +
          std::string(reply->str));
    }
    auto offset = str.find(",");
    int64_t full_version = atoll(str.substr(0, offset).c_str());
    int64_t latest_version = atoll(str.substr(offset+1).c_str());
    meta->model_version.push_back(latest_version);
    meta->curr_full_version.push_back(full_version);
  }
  freeReplyObject(reply);

  return Status::OK();
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

Status LocalRedis::GetStorageMeta(StorageMeta* meta) {
  // currently need to get 'active' and 'model_version'
  // of redis db-0 and db-1.
  // db-0
  StorageMeta tmp0;
  Status s = GetRedisMeta(c_, 0, &tmp0);
  if (!s.ok()) return s;
  meta->active.push_back(tmp0.active[0]);
  meta->model_version.push_back(tmp0.model_version[0]);

  // db-1
  StorageMeta tmp1;
  s = GetRedisMeta(c_, 1, &tmp1);
  if (!s.ok()) return s;
  meta->active.push_back(tmp1.active[0]);
  meta->model_version.push_back(tmp1.model_version[0]);

  return Status::OK();
}

Status LocalRedis::SetActiveStatus(bool active) {
  std::string set_active_cmd("SET active 0");
  if (active) {
    set_active_cmd = "SET active 1";
  }
  redisReply* reply = (redisReply *)redisCommand(
      c_, set_active_cmd.c_str());
  if (REDIS_REPLY_STATUS != reply->type) {
    freeReplyObject(reply);
    return Status(error::Code::INTERNAL,
        "[Redis] run redisCommand-SET active failed. " +
        std::string(reply->str));
  }
  freeReplyObject(reply);

  return Status::OK();
}

Status LocalRedis::GetModelVersion(int64_t* full_version,
                                   int64_t* latest_version) {
  std::string cmd("GET model_version");
  redisReply* reply = (redisReply *)redisCommand(c_, cmd.c_str());
  if (REDIS_REPLY_NIL == reply->type) {
    *full_version = -1;
    *latest_version = -1;
  } else if (REDIS_REPLY_STRING == reply->type) {
    std::string str(reply->str);
    if (str.find(",") == std::string::npos) {
      freeReplyObject(reply);
      return Status(error::Code::INTERNAL,
          "[Redis] Parse model_version failed. " +
          std::string(reply->str));
    }
    auto offset = str.find(",");
    *full_version = atoll(str.substr(0, offset).c_str());
    *latest_version = atoll(str.substr(offset+1).c_str());
  } else {
    freeReplyObject(reply);
    return Status(error::Code::INTERNAL,
        "[Redis] run redisCommand-Get model_version failed. " +
        std::string(reply->str));
  }
  freeReplyObject(reply);

  return Status::OK();
}

Status LocalRedis::SetModelVersion(int64_t full_version,
                                   int64_t latest_version) {
  std::string cmd = "SET model_version " +
                    std::to_string(full_version) +
                    "," + std::to_string(latest_version);
  redisReply* reply = (redisReply *)redisCommand(
      c_, cmd.c_str());
  if (REDIS_REPLY_STATUS != reply->type) {
    freeReplyObject(reply);
    return Status(error::Code::INTERNAL,
        "[Redis] run redisCommand-SET model_version failed. " +
        std::string(reply->str));
  }
  freeReplyObject(reply);

  return Status::OK();
}

// NOTE:(jiankeng.pt) The distributed lock only suit for
// single master case. Of course the cluster can have
// many slave nodes.
Status LocalRedis::GetStorageLock(
    int value, int timeout, bool* success) {
  *success = false;
  std::string cmd = "set model_lock " + std::to_string(value) +
                    " ex " + std::to_string(timeout) + " nx";
  redisReply* reply =
      (redisReply *)redisCommand(c_, cmd.c_str());
  if(reply->type != REDIS_REPLY_NIL &&
     string(reply->str) == "OK") {
    LOG(INFO) << "Get redis lock successful.";
    *success = true;
  }
  freeReplyObject(reply);

  return Status::OK();
}

Status LocalRedis::ReleaseStorageLock(int value) {
  char script[] = "if redis.call('get', KEYS[1]) == ARGV[1] then "
                  "return redis.call('del', KEYS[1]) else return 0 end";

  std::string value_str = std::to_string(value);
  redisReply* reply = (redisReply *)redisCommand(
      c_, "eval %s %d %s %s", script, 1,
      "model_lock", value_str.c_str());

  if (reply->type == REDIS_REPLY_INTEGER &&
      reply->integer == 1) {
    LOG(INFO) << "Release redis lock successful.";
  } else {
    LOG(INFO) << "Release redis lock failed.";
  }

  return Status::OK();
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
  freeReplyObject(r);
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

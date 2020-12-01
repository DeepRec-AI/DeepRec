#include <unistd.h>
#include <vector>
#include <string>

#include "odl_processor/model_store/redis_model_store.h"
#include "adapters/libevent.h"
#include "async.h"
#include "hiredis.h"
#include "tensorflow/core/lib/core/status.h"

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

LocalRedis::LocalRedis(Config config)
  : ip_(config.ip),
    port_(config.port),
    ac_(nullptr),
    base_(nullptr) {
  signal(SIGPIPE, SIG_IGN);
  assert((ac_ = redisAsyncConnect(ip_.c_str(), port_)) != nullptr);
  base_ = event_base_new();
  redisLibeventAttach(ac_, base_);
  redisAsyncSetConnectCallback(ac_, ConnectCallback);
  redisAsyncSetDisconnectCallback(ac_, DisconnectCallback);
  event_thread_.reset(
      new std::thread(&event_base_dispatch, base_));
}

LocalRedis::~LocalRedis() {
  LOG(INFO) << "~LocalRedis";
  redisAsyncDisconnect(ac_);
  if (ac_) {
    redisAsyncFree(ac_);
  }
  event_base_loopexit(base_, NULL);
  event_thread_->join();
}

Status LocalRedis::Cleanup() {
  struct CleanupCallbackWrapper* wrapper = new CleanupCallbackWrapper();
  int s = redisAsyncCommand(ac_, CleanupCallback, (void*)wrapper, "FLUSHDB");
  if (REDIS_OK == s) {
    while (!wrapper->done) {
      sleep(1);
    }
    LOG(INFO) << "LocalRedis::Cleanup, Done." << wrapper->s.ToString();
    Status s = wrapper->s;
    delete wrapper;
    return s;
  } else {
    delete wrapper;
    return Status(error::Code::INTERNAL,
        "[Redis] run redisAsyncCommand-FLUSHDB failed.");
  }
}

Status LocalRedis::BatchGetAsync(uint64_t feature2id,
                                 const char* const keys,
                                 char* const values,
                                 size_t bytes_per_key,
                                 size_t bytes_per_values,
                                 size_t N,
                                 const char* default_value,
                                 BatchGetCallback cb) {
  struct GetCallbackWrapper* wrapper = new GetCallbackWrapper();
  wrapper->cb = std::move(cb);
  wrapper->tensor_data = values;
  wrapper->stride_in_bytes = bytes_per_values;
  wrapper->default_value = default_value;

  int64_t len = N;
  size_t keys_byte_lens = bytes_per_key;
  char ** argv = new char*[len + 1 ];
  size_t * argvlen = new size_t[len + 1 ];

  int j = 0;
  argv[j] = new char[5];
  memcpy(argv[j],"MGET",4);
  argvlen[j] = 4;
  ++j;

  size_t key_length = keys_byte_lens + sizeof(feature2id);
  for(int i = 0 ; i < len; i++) {
    argvlen[j] = key_length;
    argv[j] = new char[key_length];
    memcpy((void*)argv[j], keys + i*keys_byte_lens, keys_byte_lens);
    memcpy((void*)(argv[j] + keys_byte_lens), &feature2id, sizeof(feature2id));
    j++;
  }
  int s = redisAsyncCommandArgv(ac_, GetCallback, (void*)wrapper,
              len + 1, const_cast<const char **>(argv), argvlen);

  for(int i = 0; i < j; i++) {
    delete [] argv[i];
    argv[i] = NULL;
  }
  delete []argv;
  delete []argvlen;

  if (REDIS_OK != s) {
    return Status(error::Code::INTERNAL,
        "[Redis] run redisAsyncCommand-MGET failed.");
  }
  return Status::OK();
}

Status LocalRedis::BatchSetAsync(uint64_t feature2id,
                                 const char* const keys,
                                 const char* const values,
                                 size_t bytes_per_key,
                                 size_t bytes_per_values,
                                 size_t N,
                                 BatchSetCallback cb) {
  struct SetCallbackWrapper* wrapper = new SetCallbackWrapper();
  wrapper->cb = std::move(cb);

  int64_t len = N;
  size_t keys_byte_lens = bytes_per_key;
  size_t values_byte_lens = bytes_per_values;
  char ** argv = new char*[2*len + 1];
  size_t * argvlen = new size_t[2*len + 1];

  int j = 0;
  argv[j] = new char[5];
  memcpy(argv[j], "MSET", 4);
  argvlen[j] = 4;
  ++j;


  size_t key_length = keys_byte_lens + sizeof(feature2id);
  for(int i = 0 ; i < len; i++) {
    argvlen[j] = key_length;
    argv[j] = new char[key_length];
    memcpy((void*)argv[j], keys + i*keys_byte_lens, keys_byte_lens);
    memcpy((void*)(argv[j] + keys_byte_lens), &feature2id, sizeof(feature2id));
    j++;

    argvlen[j] = values_byte_lens;
    argv[j] = new char[values_byte_lens];
    memcpy((void*)argv[j], values + i*values_byte_lens, values_byte_lens);
    j++;
  }

  int s = redisAsyncCommandArgv(ac_, SetCallback, (void*)wrapper,
              len * 2 + 1, const_cast<const char **>(argv), argvlen);

  for(int i = 0; i < j; i++) {
    delete [] argv[i];
    argv[i] = NULL;
  }
  delete []argv;
  delete []argvlen;

  if (REDIS_OK != s) {
    return Status(error::Code::INTERNAL,
        "[Redis] run redisAsyncCommand-MSET failed.");
  }
  return Status::OK();
}

} // namespace processor
} // namespace tensorflow

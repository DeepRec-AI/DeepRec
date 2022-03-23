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
  std::vector<char*> values;
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
      callback(s, std::vector<int64_t>());
    } else {
      Status s(error::Code::INTERNAL,
        "[Redis] run GetCallback failed: unknown error");
      callback(s, std::vector<int64_t>());
    }
  } else {
    if (reply->type == REDIS_REPLY_ARRAY) {
      std::vector<int64_t> nil_indices;
      for (int i = 0; i < reply->elements; i++) {
        if (reply->type == REDIS_REPLY_NIL) {
          nil_indices.push_back(i);
        } else if (REDIS_REPLY_STRING) {
          memcpy(wrapper->values[i], reply->element[i]->str, reply->element[i]->len);
        } else {
          Status s(error::Code::INTERNAL,
            "[Redis] run GetCallback failed.");
          callback(s, nil_indices);
          delete wrapper;
          return;
        }
      }
      Status s;
      callback(s, nil_indices);
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

Status LocalRedis::RegisterFeatures(const std::vector<std::string>& features) {
  // TODO
  return Status::OK();
}

Status LocalRedis::BatchGetAsync(const std::string& feature,
                                 const std::string& version,
                                 const std::vector<char*>& keys,
                                 size_t keys_byte_lens,
                                 const std::vector<char*>& values,
                                 BatchGetCallback cb) {
  CHECK_EQ(keys.size(), values.size());
  struct GetCallbackWrapper* wrapper = new GetCallbackWrapper();
  wrapper->cb = std::move(cb);
  wrapper->values = values;

  int64_t len = keys.size();
  char ** argv = new char*[keys.size() + 1 ];
  size_t * argvlen = new size_t[keys.size() + 1 ];

  int j = 0;
  argv[j] = new char[5];
  memcpy(argv[j],"MGET",4);
  argvlen[j] = 4;
  ++j;


  for(int i = 0 ; i < keys.size(); i++) {
    argvlen[j] = keys_byte_lens;
    argv[j] = new char[keys_byte_lens];
    memset((void*)argv[j], 0, keys_byte_lens);
    memcpy((void*)argv[j], keys[i], keys_byte_lens);
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

  if (!s) {
    return Status(error::Code::INTERNAL,
        "[Redis] run redisAsyncCommand-MGET failed.");
  }
  return Status::OK();
}

Status LocalRedis::BatchSetAsync(const std::string& feature,
                                 const std::string& version,
                                 const std::vector<char*>& keys,
                                 size_t keys_byte_lens,
                                 const std::vector<char*>& values,
                                 size_t values_byte_lens,
                                 BatchSetCallback cb) {
  CHECK_EQ(keys.size(), values.size());
  struct SetCallbackWrapper* wrapper = new SetCallbackWrapper();
  wrapper->cb = std::move(cb);

  std::string cmd = "MSET";
  int64_t len = keys.size();
  char ** argv = new char*[keys.size() + values.size() + 1];
  size_t * argvlen = new size_t[keys.size() + values.size() + 1];

  int j = 0;
  argv[j] = new char[5];
  memcpy(argv[j], "MSET", 4);
  argvlen[j] = 4;
  ++j;


  for(int i = 0 ; i < keys.size(); i++) {
    argvlen[j] = keys_byte_lens;
    argv[j] = new char[keys_byte_lens];
    memset((void*)argv[j], 0, keys_byte_lens);
    memcpy((void*)argv[j], keys[i], keys_byte_lens);
    j++;

    argvlen[j] = values_byte_lens;
    argv[j] = new char[values_byte_lens];
    memset((void*)argv[j], 0, values_byte_lens);
    memcpy((void*)argv[j], values[i], values_byte_lens);
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

  if (!s) {
    return Status(error::Code::INTERNAL,
        "[Redis] run redisAsyncCommand-MSET failed.");
  }
  return Status::OK();
}

} // namespace processor
} // namespace tensorflow

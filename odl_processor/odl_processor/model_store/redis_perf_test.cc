#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <chrono>
#include <string>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>

#include "hiredis.h"
#include "async.h"
#include "net.h"
#include "adapters/libevent.h"

#define DEBUG 0

int set = 0;

inline uint64_t GetTimeStamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>
      (std::chrono::system_clock::now().time_since_epoch()).count();
}

void getCallback(redisAsyncContext *ac, void *r, void *privdata) {
  redisReply *reply = (redisReply*)r;
  if (reply == NULL) {
    if (ac->errstr) {
      printf("errstr: %s\n", ac->errstr);
    }
    return;
  }
  --set;
  if (set == 0) {
    uint64_t start = *(uint64_t*)privdata;
    uint64_t end = GetTimeStamp();
    printf("Async Multi(batch: %d) GET %d kv, cost %.2f sec\n", -1, -1, (end-start)/1e3);
    redisAsyncDisconnect(ac);
  }
#if DEBUG
  if (reply->type == REDIS_REPLY_ARRAY) {
    for (int i = 0; i < reply->elements; i++) {
      printf("%u %s\n", i, reply->element[i]->str);
    }
  } else if (reply->type == REDIS_REPLY_STATUS) {
    printf("GET reply, %s\n", reply->str);
  } else {
    printf("ERROR!!ERROR[%s]: %d\n", (char*)privdata, reply->str);
  }
#endif
}

void setCallback(redisAsyncContext *ac, void *r, void *privdata) {
  redisReply *reply = (redisReply*)r;
  if (reply == NULL) {
    if (ac->errstr) {
      printf("errstr: %s\n", ac->errstr);
    }
    return;
  }
  ++set;
  if (set == 1000) {
    uint64_t start = *(uint64_t*)privdata;
    uint64_t end = GetTimeStamp();
    printf("Async Multi(batch: %d) SET %d kv, cost %.2f sec\n", -1, -1, (end-start)/1e3);
    redisAsyncDisconnect(ac);
  }
#if DEBUG
  if (reply->type == REDIS_REPLY_ARRAY) {
    for (int i = 0; i < reply->elements; i++) {
      printf("%u %s\n", i, reply->element[i]->str);
    }
  } else if (reply->type == REDIS_REPLY_STATUS) {
    printf("SET reply, %s\n", reply->str);
  } else {
    printf("ERROR!!ERROR[%s]: %s\n", (char*)privdata, reply->str);
  }
#endif
}

void endCallback(redisAsyncContext *ac, void *r, void *privdata) {
  redisReply *reply = (redisReply*)r;
  if (reply->type == REDIS_REPLY_STATUS) {
    printf("GET reply, %s, msg: %s\n", reply->str, (char*)privdata);
    redisAsyncDisconnect(ac);
  }
}

void connectCallback(const redisAsyncContext *c, int status) {
  if (status != REDIS_OK) {
    printf("Error: %s\n", c->errstr);
    return;
  }
  printf("Connected...\n");
}

void disconnectCallback(const redisAsyncContext *c, int status) {
  if (status != REDIS_OK) {
    printf("Error: %s\n", c->errstr);
    return;
  }
  printf("Disconnected...\n");
}

int main(int argc, char **argv) {
  printf("==============Redis Perf Test (one-DB)==============\n");
#if DEBUG
  const uint64_t total_count = 5;
#else
  const uint64_t total_count = 1000 * 1000;
#endif
  const char* str = "def";

  printf("==============0. Single get/put String ==============\n");
  {
    uint64_t t1 = GetTimeStamp();
    for (uint64_t i = 0; i < total_count; ++i) {
      std::to_string(i).c_str();
    }
    uint64_t t2 = GetTimeStamp();
    printf("ToString %d kv, cost %.2f sec\n", total_count, (t2-t1)/1e3);
  }
  printf("==============1. Single get/put Sync ==============\n");
  {
    redisContext *c;
    redisReply *reply;
    assert((c = redisConnect("127.0.0.1", 6379)) != NULL);
    uint64_t t1 = GetTimeStamp();
    for (uint64_t i = 0; i < total_count; ++i) {
      reply = (redisReply *)redisCommand(c, "SET %s %s", std::to_string(i).c_str(), str);
#if DEBUG
      printf("SET reply, %s\n", reply->str);
#endif
    }
    uint64_t t2 = GetTimeStamp();
    for (uint64_t i = 0; i < total_count; ++i) {
      reply = (redisReply *)redisCommand(c, "GET %s", std::to_string(i).c_str());
#if DEBUG
      printf("GET reply, %s\n", reply->str);
#endif
    }
    uint64_t t3 = GetTimeStamp();
    redisCommand(c, "flushall");
    printf("SET %d kv, cost %.2f sec\n", total_count, (t2-t1)/1e3);
    printf("GET %d kv, cost %.2f sec\n", total_count, (t3-t2)/1e3);
  }
  printf("==============2. Pipelining get/put Sync ==============\n");
  {
    redisContext *c;
    redisReply *reply;
    assert((c = redisConnect("127.0.0.1", 6379)) != NULL);
#if DEBUG
    const int batch = 5;
#else
    const int batch = 1000;
#endif
    int batch_id =0;
    uint64_t t1 = GetTimeStamp();
    for (uint64_t i = 0; i < total_count; ++i) {
      reply = (redisReply *)redisAppendCommand(c, "SET %s %s", std::to_string(i).c_str(), str);
      batch_id++;
      if (batch_id == batch) {
        for (; batch_id > 0; batch_id--) {
          redisReply* reply;
          redisGetReply(c, (void**)&reply);
#if DEBUG
          printf("SET reply, %s\n", reply->str);
#endif
          freeReplyObject(reply);
        }
      }
    }
    batch_id =0;
    uint64_t t2 = GetTimeStamp();
    for (uint64_t i = 0; i < total_count; ++i) {
      reply = (redisReply *)redisAppendCommand(c, "GET %s", std::to_string(i).c_str());
      batch_id++;
      if (batch_id == batch) {
        for (; batch_id > 0; batch_id--) {
          redisReply* reply;
          redisGetReply(c, (void**)&reply);
#if DEBUG
          printf("GET reply, %s\n", reply->str);
#endif
          freeReplyObject(reply);
        }
      }
    }
    uint64_t t3 = GetTimeStamp();
    redisCommand(c, "flushall");
    printf("Pipelining(batch: %d) SET %d kv, cost %.2f sec\n", batch, total_count, (t2-t1)/1e3);
    printf("Pipelining(batch: %d) GET %d kv, cost %.2f sec\n", batch, total_count, (t3-t2)/1e3);
  }
  printf("==============3. Multi get/put Sync ==============\n");
  {
    redisContext *c;
    redisReply *reply;
    assert((c = redisConnect("127.0.0.1", 6379)) != NULL);
#if DEBUG
    const int batch = 5;
#else
    const int batch = 1000;
#endif
    uint64_t t1 = GetTimeStamp();
    for (uint64_t i = 0; i < total_count/batch; ++i) {
      std::string cmd = "MSET";
      for (int j = 0; j < batch; ++j) {
        cmd += " " + std::to_string(i*batch + j) + " " + std::string(str);
      }
      reply = (redisReply *)redisCommand(c, cmd.c_str());
#if DEBUG
      printf("MSET reply, %s\n", reply->str);
#endif
    }
    uint64_t t2 = GetTimeStamp();
    for (uint64_t i = 0; i < total_count/batch; ++i) {
      std::string cmd = "MGET";
      for (int j = 0; j < batch; ++j) {
        cmd += " " + std::to_string(i*batch + j);
      }
      reply = (redisReply *)redisCommand(c, cmd.c_str());
#if DEBUG
      if (reply->type == REDIS_REPLY_ARRAY) {
        for (i = 0; i < reply->elements; i++) {
          printf("%u %s\n", i, reply->element[i]->str);
        }
      }
#endif
    }
    uint64_t t3 = GetTimeStamp();
    redisCommand(c, "flushall");
    printf("Multi(batch: %d) SET %d kv, cost %.2f sec\n", batch, total_count, (t2-t1)/1e3);
    printf("Multi(batch: %d) GET %d kv, cost %.2f sec\n", batch, total_count, (t3-t2)/1e3);
  }
  printf("==============4. Batch get/put ASync ==============\n");
  {
    signal(SIGPIPE, SIG_IGN);
    struct event_base *base = event_base_new();
    redisAsyncContext *ac;
    assert((ac = redisAsyncConnect("127.0.0.1", 6379)) != NULL);
    redisLibeventAttach(ac, base);
    redisAsyncSetConnectCallback(ac, connectCallback);
    redisAsyncSetDisconnectCallback(ac, disconnectCallback);

#if DEBUG
    const int batch = 5;
#else
    const int batch = 1000;
#endif
    uint64_t t1 = GetTimeStamp();
    for (uint64_t i = 0; i < total_count/batch; ++i) {
      std::string cmd = "MSET";
      for (int j = 0; j < batch; ++j) {
        cmd += " " + std::to_string(i*batch + j) + " " + std::string(str);
      }
      int s = redisAsyncCommand(ac, setCallback, &t1, cmd.c_str());
    }
    event_base_dispatch(base);

    assert((ac = redisAsyncConnect("127.0.0.1", 6379)) != NULL);
    redisLibeventAttach(ac, base);
    redisAsyncSetConnectCallback(ac, connectCallback);
    redisAsyncSetDisconnectCallback(ac, disconnectCallback);
    uint64_t t2 = GetTimeStamp();
    for (uint64_t i = 0; i < total_count/batch; ++i) {
      std::string cmd = "MGET";
      for (int j = 0; j < batch; ++j) {
        cmd += " " + std::to_string(i*batch + j);
      }
      int s = redisAsyncCommand(ac, getCallback, &t2, cmd.c_str());
    }
    event_base_dispatch(base);

    assert((ac = redisAsyncConnect("127.0.0.1", 6379)) != NULL);
    redisLibeventAttach(ac, base);
    redisAsyncSetConnectCallback(ac, connectCallback);
    redisAsyncSetDisconnectCallback(ac, disconnectCallback);
    uint64_t t3 = GetTimeStamp();
    redisAsyncCommand(ac, endCallback, (char*)"clean-up", "flushall");

    printf("start event dispatch\n");
    event_base_dispatch(base);
  }
  return 0;
}

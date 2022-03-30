#include <assert.h>
#include <signal.h>
#include <stdio.h>

#include "hiredis.h"
#include "async.h"
#include "net.h"
#include "adapters/libevent.h"

void connectCallback(const redisAsyncContext *c, int status) {
    if (status != REDIS_OK) {
        printf("Error: %s\n", c->errstr);
        return;
    }
    printf("Connected...\n");
}

int main(int argc, char **argv) {
#if 0
    redisContext *c;
    redisReply *reply;

    assert((c = redisConnect("127.0.0.1", 6379)) != NULL);
    reply = (redisReply *)redisCommand(c, "GET 1");
    printf("reply, %s", reply->str);
#else
    signal(SIGPIPE, SIG_IGN);
    struct event_base *base = event_base_new();
    redisOptions options = {0};
    REDIS_OPTIONS_SET_TCP(&options, "127.0.0.1", 6379);
    struct timeval tv = {0};
    tv.tv_sec = 1;
    options.connect_timeout = &tv;

    redisAsyncContext *ac;
    assert((ac = redisAsyncConnect("127.0.0.1", 6379)) != NULL);
    redisLibeventAttach(ac, base);
    redisAsyncSetConnectCallback(ac,connectCallback);
    printf("start event dispatch\n");
    event_base_dispatch(base);
    //redisAsyncFree(ac);

#endif
    return 0;
}

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "libevent",
    srcs = glob(["*.c"],
                exclude=["evthread_win32.c",
                         "win32select.c",
                         "buffer_iocp.c",
                         "event_iocp.c",
                         "bufferevent_openssl.c",
                         "bufferevent_async.c"]
             ),
    hdrs = glob(["*.h",
                 "include/*.h",
                 "include/event2/*.h",
                 "arc4random.c",]),
    includes=["include/event2",
              "include",],
    linkopts = [
        "-lpthread",
        "-ldl",
    ],
)

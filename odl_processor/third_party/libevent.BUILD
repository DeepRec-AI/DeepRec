package(default_visibility = ["//visibility:public"])

cc_library(
    name = "libevent",
    srcs = glob(["libevent-2.1.12-stable/*.c"],
                exclude=["libevent-2.1.12-stable/evthread_win32.c",
                         "libevent-2.1.12-stable/win32select.c",
                         "libevent-2.1.12-stable/buffer_iocp.c",
                         "libevent-2.1.12-stable/event_iocp.c",
                         "libevent-2.1.12-stable/bufferevent_openssl.c",
                         "libevent-2.1.12-stable/bufferevent_async.c"]
             ),
    hdrs = glob(["libevent-2.1.12-stable/*.h",
                 "libevent-2.1.12-stable/include/*.h",
                 "libevent-2.1.12-stable/include/event2/*.h",
                 "libevent-2.1.12-stable/arc4random.c",]),
    includes=["libevent-2.1.12-stable/include/event2",
              "libevent-2.1.12-stable/include",
              "libevent-2.1.12-stable"],
    linkopts = [
        "-lpthread",
        "-ldl",
    ],
)

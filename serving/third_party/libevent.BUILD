package(default_visibility = ["//visibility:public"])

cc_library(
    name = "libevent",
    srcs = glob(["libevent-release-2.1.12-stable/*.c"],
                exclude=["libevent-release-2.1.12-stable/evthread_win32.c",
                         "libevent-release-2.1.12-stable/win32select.c",
                         "libevent-release-2.1.12-stable/buffer_iocp.c",
                         "libevent-release-2.1.12-stable/event_iocp.c",
                         "libevent-release-2.1.12-stable/bufferevent_openssl.c",
                         "libevent-release-2.1.12-stable/bufferevent_async.c"]
             ),
    hdrs = glob(["libevent-release-2.1.12-stable/*.h",
                 "libevent-release-2.1.12-stable/include/*.h",
                 "libevent-release-2.1.12-stable/include/event2/*.h",
                 "libevent-release-2.1.12-stable/arc4random.c",]),
    includes=["libevent-release-2.1.12-stable/include/event2",
              "libevent-release-2.1.12-stable/include",
              "libevent-release-2.1.12-stable"],
    linkopts = [
        "-lpthread",
        "-ldl",
    ],
)

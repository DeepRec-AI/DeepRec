#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_REMOTE_WORKER_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_REMOTE_WORKER_H_

namespace seastar {
class channel;
}

namespace tensorflow {

class WorkerInterface;
class WorkerCacheLogger;
struct WorkerEnv;

WorkerInterface* NewSeastarRemoteWorker(seastar::channel* seastar_channel,
                                        WorkerCacheLogger* logger,
                                        WorkerEnv* env);

} // namespace tensorflow

#endif // TENSORFLOW_CONTIRB_STAR_SEASTAR_SEASTAR_REMOTE_WORKER_H_

#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_WORKER_CACHE_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_WORKER_CACHE_H_

#include "tensorflow/core/distributed_runtime/worker_cache.h"


namespace tensorflow {

class WorkerInterface;
class WorkerEnv;
class SeastarChannelCache;

WorkerCacheInterface* NewSeastarWorkerCache(
    SeastarChannelCache* channel_cache, WorkerEnv* env);

WorkerCacheInterface* NewSeastarWorkerCacheWithLocalWorker(
    SeastarChannelCache* channel_cache, WorkerInterface* local_worker,
    const string& local_target, WorkerEnv* env);

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_WORKER_CACHE_H_

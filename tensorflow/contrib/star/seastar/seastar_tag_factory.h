#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_TAG_FACTORY_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_TAG_FACTORY_H_

#include <functional>
#include "tensorflow/contrib/star/seastar/seastar_header.h"
#include "tensorflow/contrib/star/star_worker_service_method.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace seastar {
class channel;
}

namespace tensorflow {

class SeastarClientTag;
class SeastarServerTag;
class StarWorkerService;

class SeastarTagFactory {
public:
  explicit SeastarTagFactory(StarWorkerService* worker_service);
  virtual ~SeastarTagFactory() {}

  SeastarClientTag* CreateSeastarClientTag(
      seastar::temporary_buffer<char>& header);

  SeastarServerTag* CreateSeastarServerTag(
      seastar::temporary_buffer<char>& header,
      seastar::channel* seastar_channel);
  
private:
  StarWorkerService* worker_service_;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_TAG_FACTORY_H_

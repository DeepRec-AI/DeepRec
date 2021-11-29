#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SERVER_TAG_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SERVER_TAG_H_

#include "tensorflow/contrib/star/star_server_tag.h"

namespace seastar {
class channel;
class user_packet;
}

namespace tensorflow {

class SeastarServerTag : public StarServerTag {
 public:
  SeastarServerTag(seastar::channel* seastar_channel,
                   StarWorkerService* star_worker_service);
  virtual ~SeastarServerTag() {}
  virtual void StartResp();

 private:
  seastar::user_packet* ToUserPacket();
  std::vector<seastar::user_packet*> ToUserPacketWithTensors();

 private:
  friend class SeastarTagFactory;
  seastar::channel* seastar_channel_;
  StarWorkerService* star_worker_service_;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SEASTAR_SERVER_TAG_H_


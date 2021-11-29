#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_CLIENT_TAG_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_CLIENT_TAG_H_

#include "tensorflow/contrib/star/star_client_tag.h"

namespace seastar {
class channel;
class user_packet;
}

namespace tensorflow {

class SeastarClientTag : public StarClientTag {
 public:
  SeastarClientTag(tensorflow::StarWorkerServiceMethod method,
                   WorkerEnv* env, int resp_tesnsor_count = 0,
                   int req_tensor_count = 0); 
  void StartReq(seastar::channel* seastar_channel);

 private:
  void RetryStartReq(int retry_count, seastar::channel* seastar_channel);
  seastar::user_packet* ToUserPacket();
  std::vector<seastar::user_packet*> ToUserPacketWithTensors();
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_CLIENT_TAG_H_

#include "tensorflow/contrib/star/seastar/seastar_header.h"
#include "tensorflow/contrib/star/seastar/seastar_server_tag.h"
#include "tensorflow/contrib/star/star_worker_service.h"

namespace tensorflow {

SeastarServerTag::SeastarServerTag(seastar::channel* seastar_channel,
                                   StarWorkerService* worker_service)
  : StarServerTag(worker_service), seastar_channel_(seastar_channel) {}

void SeastarServerTag::StartResp() {

  if (IsRecvTensor()  || IsStarRunGraph()) {
    seastar_channel_->put(ToUserPacketWithTensors());
  } else {
    seastar_channel_->put(ToUserPacket());
  }
}

seastar::user_packet* SeastarServerTag::ToUserPacket() {
  seastar::net::fragment respHeader {resp_header_buf_.data_, resp_header_buf_.len_};
  seastar::net::fragment respBody {resp_body_buf_.data_, resp_body_buf_.len_};

  std::vector<seastar::net::fragment> frags = { respHeader, respBody };
  seastar::user_packet* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = [this](){ this->SendRespDone(); };

  return up;
}

std::vector<seastar::user_packet*> SeastarServerTag::ToUserPacketWithTensors() {
  int64 total_len = 0;
  std::vector<seastar::user_packet*> ret;
  auto up = new seastar::user_packet;

  std::vector<seastar::net::fragment> frags;
  frags.emplace_back(seastar::net::fragment {resp_header_buf_.data_,
      resp_header_buf_.len_});
  total_len += resp_header_buf_.len_;

  if (IsStarRunGraph() || status_ != 0) {
    frags.emplace_back(seastar::net::fragment {resp_body_buf_.data_,
          resp_body_buf_.len_});
    total_len += resp_body_buf_.len_;
  }

  // For fuse recv / zero copy run graph, if error happens 'resp_tensor_count_'
  // is zero as when it is inited, so no tensor can be sent.
  for (auto i = 0; i < resp_tensor_count_; ++i) {
    frags.emplace_back(seastar::net::fragment {resp_message_bufs_[i].data_,
        resp_message_bufs_[i].len_});
    total_len += resp_message_bufs_[i].len_;

    if (resp_tensor_bufs_[i].len_ > 0) {
      frags.emplace_back(seastar::net::fragment {resp_tensor_bufs_[i].data_,
        resp_tensor_bufs_[i].len_});
      total_len += resp_tensor_bufs_[i].len_;
    }
  }
  up->_fragments = frags;
  up->_done = [this]() { this->SendRespDone(); };
  ret.emplace_back(up);

  return ret;
}

} // namespace tensorflow

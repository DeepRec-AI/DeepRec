#include "tensorflow/contrib/star/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/star/seastar/seastar_header.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {

// Default connection timeout is 120s.
static const size_t kMaxConnectionTimeoutInMS = 120000;
static const size_t kUSleepInMS = 100;
static const size_t kUSleepInUs = 1000 * kUSleepInMS;

int64 GetMaxConnectionTimeout() {
  int64 max_connection_timeout = kMaxConnectionTimeoutInMS;
  ReadInt64FromEnvVar("NETWORK_MAX_CONNECTION_TIMEOUT",
      kMaxConnectionTimeoutInMS, &max_connection_timeout);

  return max_connection_timeout;
}

std::string GenErrorMsg(bool is_init, bool is_broken, const std::string& addr) {
  std::string msg = "Seastar channel: unknown error. connection is : " + addr;

  if (!is_init) {
    msg = "Seastar channel: connection is timeout. connection is : " + addr;
  }

  if (is_broken) {
    msg = "Seastar channel: connection is broken. connection is : " + addr;
  }

  return msg;
}

} // namespace

SeastarClientTag::SeastarClientTag(tensorflow::StarWorkerServiceMethod method,
                                   WorkerEnv* env, int resp_tesnsor_count,
                                   int req_tensor_count)
  : StarClientTag(method, env, resp_tesnsor_count, req_tensor_count) {}

void SeastarClientTag::RetryStartReq(int retry_count,
                                     seastar::channel* seastar_channel) {
  usleep(kUSleepInUs);
  bool is_init = seastar_channel->is_init();
  bool is_broken = seastar_channel->is_channel_broken();

  if (is_init && !is_broken) {
    // Good case.
    LOG(WARNING) << "Seastar conn success after retry.";
    if (IsStarRunGraph()) {
      seastar_channel->put(ToUserPacketWithTensors());
    } else {
      seastar_channel->put(ToUserPacket());
    }

  } else if (--retry_count > 0) {
    // Bad case and need retry.
    LOG(WARNING) << "Seastar conn timeout for: " << seastar_channel->get_addr()
                 << ", left retry count: " << retry_count;
    ScheduleProcess([this, retry_count, seastar_channel]() {
        RetryStartReq(retry_count, seastar_channel);
      });

  } else {
    // Bad case and retry count is exhausted.
    LOG(ERROR) << "Seastar conn timeout for: " << seastar_channel->get_addr()
               << ", retry count is exhausted.";
    Status s(error::INTERNAL,
             GenErrorMsg(is_init, is_broken, seastar_channel->get_addr()));
    ScheduleProcess([this, s]() {
        HandleResponse(s);
      });
  }
}

void SeastarClientTag::StartReq(seastar::channel* seastar_channel) {
  if (seastar_channel->is_channel_broken()) {
    seastar_channel->reconnect();
  }

  bool is_init = seastar_channel->is_init();
  bool is_broken = seastar_channel->is_channel_broken();

  if (is_init && !is_broken) {
    // Good case.
    if (IsStarRunGraph()) {
      seastar_channel->put(ToUserPacketWithTensors());
    } else {
      seastar_channel->put(ToUserPacket());
    }
  } else if (!fail_fast_) {
    // Bad case and need retry.
    int max_retry = GetMaxConnectionTimeout() / kUSleepInMS;
    if (timeout_in_ms_ != 0) {
      max_retry = timeout_in_ms_ / kUSleepInMS;
    }

    // NOTE(rangeng.llb): Maybe this is in seastar thread, retry by schedule
    // again.
    // Refer to: https://workitem.aone.alibaba-inc.com/issue/16534619 for more details.
    LOG(WARNING) << "Seastar conn timeout for: " << seastar_channel->get_addr()
                 << ", now do retry with max retry count: " << max_retry;
    ScheduleProcess([this, max_retry, seastar_channel]() {
        RetryStartReq(max_retry, seastar_channel);
      });

  } else {
    // Bad case and fail fast.
    LOG(ERROR) << "Seastar conn timeout for: " << seastar_channel->get_addr()
               << ", now fail fast.";
    tensorflow::Status s(error::INTERNAL,
                         GenErrorMsg(is_init, is_broken, seastar_channel->get_addr()));
    ScheduleProcess([this, s] {
        HandleResponse(s);
      });
  }
}

seastar::user_packet* SeastarClientTag::ToUserPacket() {
  seastar::net::fragment reqHeader {req_header_buf_.data_, req_header_buf_.len_};
  seastar::net::fragment reqBody {req_body_buf_.data_, req_body_buf_.len_};

  std::vector<seastar::net::fragment> frags = { reqHeader, reqBody };
  seastar::user_packet* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = []{};

  return up;
}

std::vector<seastar::user_packet*> SeastarClientTag::ToUserPacketWithTensors() {
  int64 total_len = 0;
  std::vector<seastar::user_packet*> ret;
  auto up = new seastar::user_packet;
  std::vector<seastar::net::fragment> frags;

  frags.emplace_back(seastar::net::fragment {req_header_buf_.data_,
        req_header_buf_.len_});
  total_len += req_header_buf_.len_;

  frags.emplace_back(seastar::net::fragment {req_body_buf_.data_,
        req_body_buf_.len_});
  total_len += req_body_buf_.len_;

  for (int i = 0; i < req_tensor_count_; ++i) {
    frags.emplace_back(seastar::net::fragment {req_message_bufs_[i].data_,
          req_message_bufs_[i].len_});
    total_len += req_message_bufs_[i].len_;

    if (req_tensor_bufs_[i].len_ > 0) {
      frags.emplace_back(seastar::net::fragment {req_tensor_bufs_[i].data_,
            req_tensor_bufs_[i].len_});
      total_len += req_tensor_bufs_[i].len_;
    }
  }

  up->_fragments = frags;
  up->_done = []() {}; // no need to delete now
  ret.emplace_back(up);

  return ret;
}

} // namespace tensorflow

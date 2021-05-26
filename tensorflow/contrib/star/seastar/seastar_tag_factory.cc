#include "seastar/core/channel.hh"
#include "tensorflow/contrib/star/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/star/seastar/seastar_server_tag.h"
#include "tensorflow/contrib/star/seastar/seastar_tag_factory.h"
#include "tensorflow/contrib/star/star_message.h"
#include "tensorflow/contrib/star/star_tensor_coding.h"
#include "tensorflow/contrib/star/star_worker_service.h"


namespace tensorflow {

SeastarTagFactory::SeastarTagFactory(StarWorkerService* worker_service) :
  worker_service_(worker_service) {}

SeastarClientTag* SeastarTagFactory::CreateSeastarClientTag(
    seastar::temporary_buffer<char>& header) {
  char* p = const_cast<char*>(header.get());

  // Igonre the BBBB and method segment.
  SeastarClientTag* tag = NULL;
  memcpy(&tag, p + StarClientTag::kTagIndex, 8);
  memcpy(&tag->status_, p + StarClientTag::kStatusIndex, 4);

  if (tag->status_ != 0) {
    memcpy(&tag->err_msg_len_, p + StarClientTag::kPayloadLenIndex, 8);
    return tag;
  }

  if (tag->IsStarRunGraph()) {
    int32 meta_len = 0;
    memcpy(&meta_len, p + StarClientTag::kUserDataIndex, 4);

    tag->resp_body_buf_.len_ = meta_len;
    tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_];

  } else if (!tag->IsRecvTensor()) {
    memcpy(&tag->resp_body_buf_.len_, p + StarClientTag::kPayloadLenIndex, 8);
    tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_];
  }

  return tag;
}

SeastarServerTag* SeastarTagFactory::CreateSeastarServerTag(
    seastar::temporary_buffer<char>& header,
    seastar::channel* seastar_channel) {
  char* p = const_cast<char*>(header.get());
  SeastarServerTag* tag = new SeastarServerTag(seastar_channel,
                                               worker_service_);
  // Ignore the BBBB segment.
  memcpy(&tag->method_, p + StarClientTag::kMethodIndex, 4);
  memcpy(&tag->client_tag_id_, p + StarClientTag::kTagIndex, 8);
  // Igonre the status segment

  if (tag->IsStarRunGraph()) {
    int32 meta_len = 0;
    memcpy(&meta_len, p + StarClientTag::kUserDataIndex, 4);

    tag->req_body_buf_.len_ = meta_len;
    tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_];

  } else {
    memcpy(&(tag->req_body_buf_.len_), p + StarClientTag::kPayloadLenIndex, 8);
    tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_];
  }

  return tag;
}

} // namespace tensorflow

#include "tensorflow/contrib/star/star_message.h"
#include "tensorflow/contrib/star/star_server_tag.h"
#include "tensorflow/contrib/star/star_tensor_coding.h"
#include "tensorflow/contrib/star/star_worker_service.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/platform/logging.h"


namespace tensorflow {

void InitStarServerTag(protobuf::Message* request,
                       protobuf::Message* response,
                       StarServerTag* tag) {
  request->ParseFromArray(tag->req_body_buf_.data_,
                          tag->req_body_buf_.len_);

  StatusCallback done = [response, tag] (const Status& s) {
    tag->resp_header_buf_.len_ = StarServerTag::kHeaderSize;
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBB", 4);
    // Ingore method segment.
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kTagIndex,
           &tag->client_tag_id_, 8);
    tag->status_ = static_cast<int32>(s.code());
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kStatusIndex,
           &tag->status_, 4);
    // Ingore user data segment.

    if (s.ok()) {
      tag->resp_body_buf_.len_ = response->ByteSize();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      response->SerializeToArray(tag->resp_body_buf_.data_, tag->resp_body_buf_.len_);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &tag->resp_body_buf_.len_, 8);

    } else {
      //TODO: RemoteWorker::LoggingRequest doesn't need to response.
      //      can be more elegant.
      // Send err msg back to client

      tag->resp_body_buf_.len_ = s.error_message().length();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      memcpy(tag->resp_body_buf_.data_, s.error_message().c_str(),
             tag->resp_body_buf_.len_);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &tag->resp_body_buf_.len_, 8);
    }

    tag->StartResp();
  };

  tag->send_resp_ = std::move(done);
  tag->clear_ = [](const Status& s) {};
}

void InitStarServerTag(protobuf::Message* request,
                       StarTensorResponse* response,
                       StarServerTag* tag,
                       StatusCallback clear) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [request, response, tag] (const Status& s) {
    tag->resp_header_buf_.len_ = StarServerTag::kHeaderSize;
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBB", 4);
    // Ingore method segment.
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kTagIndex,
           &tag->client_tag_id_, 8);
    tag->status_ = static_cast<int32>(s.code());
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kStatusIndex,
           &tag->status_, 4);
    // Ingore user data segment.

    if (s.ok()) {
      tag->InitResponseTensorBufs(1);
      uint64_t payload_len
        = StarMessage::SerializeTensorMessage(response->GetTensor(),
                                              response->GetTensorProto(),
                                              response->GetIsDead(),
                                              &tag->resp_message_bufs_[0],
                                              &tag->resp_tensor_bufs_[0]);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &payload_len, 8);

    } else {
      tag->resp_body_buf_.len_ = s.error_message().length();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      memcpy(tag->resp_body_buf_.data_, s.error_message().c_str(),
             tag->resp_body_buf_.len_);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &tag->resp_body_buf_.len_, 8);
    }

    tag->StartResp();
  };

  // used for zero copy sending tensor
  tag->send_resp_ = std::move(done);
  tag->clear_ = std::move(clear);
}

void InitStarServerTag(protobuf::Message* request,
                       StarFuseTensorResponse* response,
                       StarServerTag* tag,
                       StatusCallback clear) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [request, response, tag] (const Status& s) {
    tag->resp_header_buf_.len_ = StarServerTag::kHeaderSize;
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBB", 4); 
    // Ingore method segment.
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kTagIndex,
           &tag->client_tag_id_, 8); 
    tag->status_ = static_cast<int32>(s.code());
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kStatusIndex,
           &tag->status_, 4); 
    // Ingore user data segment.

    if (s.ok()) {
      tag->InitResponseTensorBufs(response->GetFuseCount());
      uint64_t payload_len = 0;
      for (int idx = 0; idx < tag->resp_tensor_count_; ++idx) {
        payload_len
          += StarMessage::SerializeTensorMessage(response->GetTensorByIndex(idx),
                                                 response->GetTensorProtoByIndex(idx),
                                                 response->GetIsDeadByIndex(idx),
                                                 &tag->resp_message_bufs_[idx],
                                                 &tag->resp_tensor_bufs_[idx]);
      }
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &payload_len, 8);
    } else {
      tag->resp_body_buf_.len_ = s.error_message().length();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      memcpy(tag->resp_body_buf_.data_, s.error_message().c_str(),
             tag->resp_body_buf_.len_);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &tag->resp_body_buf_.len_, 8);
    }

    tag->StartResp();
  };

  // used for zero copy sending tensor, unref Tensor object after star send done
  tag->clear_ = std::move(clear);
  tag->send_resp_ = std::move(done);
}

void InitStarServerTag(StarServerTag* tag) {
  ParseMetaDataCallback wrapper_parse_meta_data
    = [tag] (const char* buf, size_t len) {
      tag->star_graph_request_.DecodeRequest(buf, len);
      tag->InitRequestTensorBufs((uint64_t)(tag->star_graph_request_.feed_names_.size()));
      return Status();
  };
  tag->parse_meta_data_ = std::move(wrapper_parse_meta_data);

  ParseMessageCallback wrapper_parse_message
    = [tag](int idx, const char* tensor_msg, size_t len) {
    CHECK_EQ(StarMessage::kMessageTotalBytes, len);
    StarMessage sm;
    StarMessage::DeserializeMessage(&sm, tensor_msg);
    tag->star_graph_request_.is_dead_.push_back(sm.is_dead_);
    tag->star_graph_request_.data_type_.push_back(sm.data_type_);
    bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);
    if (can_memcpy) {
      //TODO: Implement GPU device here
      Tensor val(cpu_allocator(), sm.data_type_, sm.tensor_shape_);
      tag->req_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&val));
      tag->req_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
      tag->req_tensor_bufs_[idx].owned_ = false;
      tag->star_graph_request_.feed_tensors_[idx] = val;
    } else {
      tag->req_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
      tag->req_tensor_bufs_[idx].data_ = new char[tag->req_tensor_bufs_[idx].len_]();
    }
    return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  ParseTensorCallback wrapper_parse_tensor
    = std::bind([tag] () {
    uint64_t count = tag->req_tensor_count_;
    for (uint64_t i = 0; i < count; ++i) {
      bool can_memcpy = DataTypeCanUseMemcpy(tag->star_graph_request_.data_type_[i]);
      if (can_memcpy) {
        //TODO: Implement GPU device here
      } else {
        TensorProto tensor_proto;
        ParseProtoUnlimited(&tensor_proto,
                            tag->req_tensor_bufs_[i].data_,
                            tag->req_tensor_bufs_[i].len_);
        Rendezvous::ParsedKey parsed_key;
        Device* device = nullptr;
        Status s = Rendezvous::ParseKey(tag->star_graph_request_.feed_names_[i], &parsed_key);
        tag->star_worker_service_->GetWorker()->env() \
           ->device_mgr->LookupDevice(parsed_key.src_device, &device);
        if (device == nullptr) {
          LOG(FATAL) << "Not found device, feed name is : " << tag->star_graph_request_.feed_names_[i];
        }

        Tensor val;
        Status status = device->MakeTensorFromProto(
          tensor_proto, AllocatorAttributes(), &val);
        tag->star_graph_request_.feed_tensors_[i] = val;
      }
    }

    return Status();
  });
  tag->parse_tensor_ = std::move(wrapper_parse_tensor);

  StatusCallback done = [tag] (const Status& s) {
    tag->resp_header_buf_.len_ = StarServerTag::kHeaderSize;
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBB", 4);
    // Ingore method segment.
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kTagIndex,
           &tag->client_tag_id_, 8);
    tag->status_ = static_cast<int32>(s.code());
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kStatusIndex,
           &tag->status_, 4);

    if (s.ok()) {
      tag->FillRespBody();
    } else {
      tag->resp_body_buf_.len_ = s.error_message().length();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      memcpy(tag->resp_body_buf_.data_, s.error_message().c_str(),
             tag->resp_body_buf_.len_);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &tag->resp_body_buf_.len_, 8);
    }

    tag->StartResp();
  };

  tag->send_resp_ = std::move(done);
  tag->clear_ = [](const Status& s) {};
}

void StarServerTag::FillRespBody() {
  uint64_t fetch_count = star_graph_response_.fetch_tensors_.size();
  int32 meta_len = fetch_count * sizeof(FetchNameLenType);
  for (uint64_t i = 0; i < fetch_count; ++i) {
    meta_len += star_graph_response_.fetch_names_[i].length();
  }
  resp_body_buf_.len_ = meta_len;
  resp_body_buf_.data_ = new char[resp_body_buf_.len_]();

  uint64_t offset = 0;
  for (uint64_t i = 0; i < fetch_count; ++i) {
    FetchNameLenType len = star_graph_response_.fetch_names_[i].length();
    memcpy(resp_body_buf_.data_ + offset, &len, sizeof(FetchNameLenType));
    offset += sizeof(FetchNameLenType);
    memcpy(resp_body_buf_.data_ + offset,
           star_graph_response_.fetch_names_[i].c_str(), len);
    offset += len;
  }
  // User used data segment to store meta len.
  memcpy(resp_header_buf_.data_ + StarServerTag::kUserDataIndex,
         &meta_len, 4);

  InitResponseTensorBufs(fetch_count);

  uint64_t payload_len = meta_len;
  for (uint64_t i = 0; i < fetch_count; ++i) {
    TensorProto tensor_proto;
    if (!DataTypeCanUseMemcpy(star_graph_response_.fetch_tensors_[i].dtype())) {
      star_graph_response_.fetch_tensors_[i].AsProtoTensorContent(&tensor_proto);
    }
    payload_len
      += StarMessage::SerializeTensorMessage(star_graph_response_.fetch_tensors_[i],
                                             tensor_proto,
                                             star_graph_response_.is_dead_[i],
                                             &resp_message_bufs_[i],
                                             &resp_tensor_bufs_[i]);
  }

  memcpy(resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
         &payload_len, 8);
}

Status StarServerTag::ParseTensor() {
  return parse_tensor_();
}

Status StarServerTag::ParseMessage(
    int idx, const char* tensor_msg, size_t len) {
  return parse_message_(idx, tensor_msg, len);
}

Status StarServerTag::ParseMetaData(const char* buf, size_t len) {
  return parse_meta_data_(buf, len);
}

int StarServerTag::GetReqTensorCount() {
  return req_tensor_count_;
}

bool StarServerTag::IsStarRunGraph() {
  return method_ ==
      StarWorkerServiceMethod::kStarRunGraph;
}

void StarServerTag::InitRequestTensorBufs(int count) {
  req_tensor_count_ = count;
  req_tensor_bufs_.resize(count);
}

uint64_t StarServerTag::GetRequestTensorSize(int idx) {
  return req_tensor_bufs_[idx].len_;
}

char* StarServerTag::GetRequestTensorBuffer(int idx) {
  return req_tensor_bufs_[idx].data_;
}

int64 StarServerTag::GetRtt() {
  timeval end_time_ts;
  gettimeofday(&end_time_ts, nullptr);
  return (end_time_ts.tv_sec * 1000 * 1000 + end_time_ts.tv_usec) -
         (start_time_ts_.tv_sec * 1000 * 1000 + start_time_ts_.tv_usec);
}

StarServerTag::StarServerTag(StarWorkerService* star_worker_service)
  : method_(StarWorkerServiceMethod::kInvalid),
    client_tag_id_(0),
    status_(0),
    req_tensor_count_(0),
    resp_tensor_count_(0),
    parse_meta_data_(nullptr),
    parse_message_(nullptr),
    parse_tensor_(nullptr),
    send_resp_(nullptr),
    clear_(nullptr),
    star_worker_service_(star_worker_service) {
  gettimeofday(&start_time_ts_, nullptr);
}

StarServerTag::~StarServerTag() {
  delete [] req_body_buf_.data_;
  delete [] resp_header_buf_.data_;
  delete [] resp_body_buf_.data_;
  
  for (int i = 0; i < resp_tensor_count_; ++i) {
    delete [] resp_message_bufs_[i].data_;
    if (resp_tensor_bufs_[i].owned_) {
      delete [] resp_tensor_bufs_[i].data_;
    }
  }

  for (uint64_t i = 0; i < req_tensor_count_; ++i) {
    if (req_tensor_bufs_[i].owned_) {
      delete [] req_tensor_bufs_[i].data_;
    }
  }
}

void StarServerTag::InitResponseTensorBufs(int32_t resp_tensor_count) {
  resp_tensor_count_ = resp_tensor_count;
  resp_message_bufs_.resize(resp_tensor_count);
  resp_tensor_bufs_.resize(resp_tensor_count);
}

bool StarServerTag::IsRecvTensor() {
  return method_ == StarWorkerServiceMethod::kRecvTensor ||
         method_ == StarWorkerServiceMethod::kFuseRecvTensor;
}

// Called by star engine, call the handler.
void StarServerTag::RecvReqDone(Status s) {
  if (!s.ok()) {
    this->send_resp_(s);
    // TODO(handle clear)
    return;
  }

  HandleRequestFunction handle = star_worker_service_->GetHandler(method_);
  (star_worker_service_->*handle)(this);
}

// Called by star engine.
void StarServerTag::SendRespDone() {
  clear_(Status());
  delete this;
}

// called when request has been processed, mainly serialize resp to wire-format,
// and send response
void StarServerTag::ProcessDone(Status s) {
  //LOG(INFO) << "enter starServerTag::ProcessDone";
  send_resp_(s);
}

uint64_t StarServerTag::GetRequestBodySize() {
  return req_body_buf_.len_;
}

char* StarServerTag::GetRequestBodyBuffer() {
  return req_body_buf_.data_;
}

} // namespace tensorflow

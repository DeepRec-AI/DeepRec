#include "tensorflow/contrib/star/star_client_tag.h"
#include "tensorflow/contrib/star/star_message.h"
#include "tensorflow/contrib/star/star_server_tag.h"
#include "tensorflow/contrib/star/star_tensor_coding.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#endif // GOOGLE_CUDA
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"


namespace tensorflow {

void InitStarClientTag(protobuf::Message* request,
                       protobuf::Message* response,
                       StatusCallback done,
                       StarClientTag* tag,
                       CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = StarClientTag::kHeaderSize;
  tag->req_header_buf_.data_ = new char[StarClientTag::kHeaderSize]();

  memcpy(tag->req_header_buf_.data_, "AAAA", 4);
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kMethodIndex,
         &tag->method_, 4);
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kTagIndex, &tag, 8);
  // Ignore the status and user_data segment.
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kPayloadLenIndex,
         &tag->req_body_buf_.len_, 8);

  StatusCallback wrapper_done
    = std::bind([response, tag](StatusCallback done,
                                const Status& s) {
                  // internal error, that we dont need to parse the response,
                  // reponse is nullptr
                  if (s.code() != error::INTERNAL) {
                    response->ParseFromArray(tag->resp_body_buf_.data_,
                                            tag->resp_body_buf_.len_);
                  }
                  if (!s.ok()) {
                    if (tag->method_ == StarWorkerServiceMethod::kLogging ||
                        tag->method_ == StarWorkerServiceMethod::kTracing) {
                      // Logging & Tracing in worker.cc is UNIMPLEMENTED, ignore the error
                    } else {
                      // Debugging info
                      LOG(INFO) << "RPC's status is not ok. status code=" << s.code()
                                << ", err msg=" << s.error_message().c_str();
                    }
                  }
                  done(s);
                  delete tag;
                },
                std::move(done),
                std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  tag->ProcessCallOptions();
}

void InitStarClientTag(protobuf::Message* request,
                       StarTensorResponse* response,
                       StatusCallback done,
                       StarClientTag* tag,
                       CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_,
                            tag->req_body_buf_.len_);


  tag->req_header_buf_.len_ = StarClientTag::kHeaderSize;
  tag->req_header_buf_.data_ = new char[StarClientTag::kHeaderSize]();

  memcpy(tag->req_header_buf_.data_, "AAAA", 4);
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kMethodIndex,
         &tag->method_, 4);
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kTagIndex, &tag, 8);
  // Ignore the status and user_data segment.
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kPayloadLenIndex,
         &tag->req_body_buf_.len_, 8);

  ParseMessageCallback wrapper_parse_message
    = [request, response, tag] (int idx, const char* tensor_msg, size_t len) {
      CHECK_EQ(StarMessage::kMessageTotalBytes, len);
      StarMessage sm;
      StarMessage::DeserializeMessage(&sm, tensor_msg);

      response->SetIsDead(sm.is_dead_);
      response->SetDataType(sm.data_type_);
      bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

      if (can_memcpy) {
        if (response->GetDevice()->tensorflow_gpu_device_info() &&
            (!response->GetOnHost())) {
 #if GOOGLE_CUDA
          // LOG(INFO) << "parse msg, can memcpy and on GPU";
          // dst tensor on gpu
          Allocator* alloc = GPUProcessState::singleton()->GetGpuHostAllocator(0);
          Tensor cpu_copy(alloc, sm.data_type_, sm.tensor_shape_);

          tag->resp_tensor_bufs_[idx].data_ =
              reinterpret_cast<char*>(DMAHelper::base(&cpu_copy));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;

          response->SetTensor(cpu_copy);
#else
          return errors::Internal("No GPU device in process");
#endif

        } else {
          // LOG(INFO) << "parse msg for no fuse, can memcpy and on cpu"
          //          << ",request:" << request->DebugString();
          Tensor val(response->GetAlloc(), sm.data_type_, sm.tensor_shape_);
          tag->resp_tensor_bufs_[idx].data_ =
              reinterpret_cast<char*>(DMAHelper::base(&val));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;

          response->SetTensor(val);
        }
      } else {
        // LOG(INFO) << "parse msg, could not memcpy, tensor bytes: " << sm.tensor_bytes_
        //          << ",request:" << request->DebugString();
        tag->resp_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
        tag->resp_tensor_bufs_[idx].data_ =
            new char[tag->resp_tensor_bufs_[idx].len_]();
      }

      return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done
    = std::bind([response, tag](StatusCallback done,
                                const Status& s) {
                  if (!s.ok()) {
                    LOG(ERROR) << "wrapper_done, status not ok. status code=" << s.code()
                               << ", err msg=" << s.error_message().c_str();
                    done(s);
                    delete tag;
                    return;
                  }

                  bool can_memcpy = DataTypeCanUseMemcpy(response->GetDataType());
                  if (can_memcpy) {
                    if (response->GetDevice()->tensorflow_gpu_device_info() &&
                        (!response->GetOnHost())) {
#if GOOGLE_CUDA
                      Tensor* gpu_copy =
                          new Tensor(response->GetAlloc(),
                                     response->GetTensor().dtype(),
                                     response->GetTensor().shape());
                      GPUUtil::CopyCPUTensorToGPU(
                          &response->GetTensor(),
                          response->GetDevice()->tensorflow_gpu_device_info()->default_context,
                          response->GetDevice(),
                          gpu_copy,
                          [gpu_copy, response, done, tag](const Status& s) {
                            CHECK(s.ok()) << "copy tensor to gpu sync";
                            response->SetTensor(*gpu_copy);
                            done(s);
                            delete gpu_copy;
                            delete tag;
                          },
                          true);
#else
                      done(errors::Internal("No GPU device in process"));
                      delete tag;
#endif
                    } else {
                      // LOG(INFO) << "wrapper_done for no fuse, nothon to do,"
                      // "in the case that tensor on cpu and can memcpy";
                      done(s);
                      delete tag;
                    }
                  } else {
                    // could not memcoy
                    // LOG(INFO) << "wrapper_done, could not memcpy, recv bytes: "
                    // << tag->resp_tensor_bufs_[0].len_
                    // << ", DataType: " << response->GetDataType();
                    ParseProtoUnlimited(&response->GetTensorProto(),
                                        tag->resp_tensor_bufs_[0].data_,
                                        tag->resp_tensor_bufs_[0].len_);
                    Tensor val;
                    Status status = response->GetDevice()->MakeTensorFromProto(
                        response->GetTensorProto(),
                        response->GetAllocAttributes(),
                        &val);
                    //LOG(INFO) << "parse msg status: " << status.error_message();
                    CHECK(status.ok()) << "make cpu tensor from proto.";
                    response->SetTensor(val);
                    done(status);
                    delete tag;
                  }
                },
                std::move(done),
                std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  tag->ProcessCallOptions();
}

void InitStarClientTag(protobuf::Message* request,
                       StarFuseTensorResponse* response,
                       StatusCallback done,
                       StarClientTag* tag,
                       CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = StarClientTag::kHeaderSize;
  tag->req_header_buf_.data_ = new char[StarClientTag::kHeaderSize]();

  memcpy(tag->req_header_buf_.data_, "AAAA", 4); 
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kMethodIndex,
         &tag->method_, 4); 
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kTagIndex, &tag, 8); 
  // Ignore the status and user_data segment.
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kPayloadLenIndex,
         &tag->req_body_buf_.len_, 8); 

  ParseMessageCallback wrapper_parse_message
    = [request, response, tag] (int idx, const char* tensor_msg, size_t len) {
      CHECK_EQ(StarMessage::kMessageTotalBytes, len);
      StarMessage sm; 
      StarMessage::DeserializeMessage(&sm, tensor_msg);

      response->SetIsDeadByIndex(idx, sm.is_dead_);
      response->SetDataTypeByIndex(idx, sm.data_type_);
      bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

      if (can_memcpy) {
        if (response->GetDevice()->tensorflow_gpu_device_info() &&
            (!response->GetOnHost())) {
 #if GOOGLE_CUDA
          Allocator* alloc = GPUProcessState::singleton()->GetGpuHostAllocator(0);
          Tensor cpu_copy(alloc, sm.data_type_, sm.tensor_shape_);

          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&cpu_copy));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;

          response->SetTensorByIndex(idx, cpu_copy);
#else
          return errors::Internal("No GPU device in process");
#endif

        } else {
          Tensor val(response->GetAlloc(), sm.data_type_, sm.tensor_shape_);
          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&val));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;

          response->SetTensorByIndex(idx, val);
        }
      } else {
        tag->resp_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
        tag->resp_tensor_bufs_[idx].data_ = new char[tag->resp_tensor_bufs_[idx].len_]();
      }

      return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done
    = std::bind([response, tag](StatusCallback done,
                                const Status& s) {
                  if (!s.ok()) {
                    LOG(ERROR) << "wrapper_done, status not ok. status code=" << s.code()
                               << ", err msg=" << s.error_message().c_str();
                    done(s);
                    delete tag;
                    return;
                  }

                  int resp_tensor_count = tag->resp_tensor_count_;
                  int *resp_tensor_counter = new int(resp_tensor_count);

                  for (int idx = 0; idx < resp_tensor_count; ++idx) {
                    bool can_memcpy = DataTypeCanUseMemcpy(response->GetDataTypeByIndex(idx));
                    if (can_memcpy) {
                      if (response->GetDevice()->tensorflow_gpu_device_info() &&
                          (!response->GetOnHost())) {
#if GOOGLE_CUDA
                        Tensor* gpu_copy = new Tensor(response->GetAlloc(),
                                                      response->GetTensorByIndex(idx).dtype(),
                                                      response->GetTensorByIndex(idx).shape());
                        GPUUtil::CopyCPUTensorToGPU(&response->GetTensorByIndex(idx),
                                                    response->GetDevice()->tensorflow_gpu_device_info()->default_context,
                                                    response->GetDevice(),
                                                    gpu_copy,
                                                    [gpu_copy, response, done, tag, resp_tensor_counter, idx](const Status& s) {
                                                      CHECK(s.ok()) << "copy tensor to gpu sync";
                                                      response->SetTensorByIndex(idx, *gpu_copy);
                                                      delete gpu_copy;
                                                      if (__sync_sub_and_fetch(resp_tensor_counter, 1) == 0) {
                                                        delete resp_tensor_counter;
                                                        done(s);
                                                        delete tag;
                                                      }
                                                    },
                                                    true);
#else
                        done(errors::Internal("No GPU device in process"));
                        // delete tag;
                        // It may be not safe to delete tag here, just abort here.
                        abort();
#endif
                      } else {
                        if (__sync_sub_and_fetch(resp_tensor_counter, 1) == 0) {
                          delete resp_tensor_counter;
                          done(s);
                          delete tag;
                        }
                      }
                    } else {
                      // Could not memory copy.
                      ParseProtoUnlimited(&response->GetTensorProtoByIndex(idx),
                                          tag->resp_tensor_bufs_[idx].data_,
                                          tag->resp_tensor_bufs_[idx].len_);
                      Tensor val;
                      Status status = response->GetDevice()->MakeTensorFromProto(
                          response->GetTensorProtoByIndex(idx),
                          response->GetAllocAttributes(), &val);
                      CHECK(status.ok()) << "Make cpu tensor from proto.";
                      response->SetTensorByIndex(idx, val);
                      if (__sync_sub_and_fetch(resp_tensor_counter, 1) == 0) {
                        delete resp_tensor_counter;
                        done(status);
                        delete tag;
                      }
                    }
                  } // End for loop of the fuse count.
                },
                std::move(done),
                std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  tag->ProcessCallOptions();
}

void InitStarClientTag(StarRunGraphRequest* request,
                       StarRunGraphResponse* response,
                       StatusCallback done,
                       StarClientTag* tag) {
  request->EncodeRequest(&tag->req_body_buf_);

  tag->req_header_buf_.len_ = StarClientTag::kHeaderSize;
  tag->req_header_buf_.data_ = new char[StarClientTag::kHeaderSize]();

  memcpy(tag->req_header_buf_.data_, "AAAA", 4);
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kMethodIndex,
         &tag->method_, 4);
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kTagIndex, &tag, 8);
  // Ignore the status segment.
  // Use the user_data segment to store the meta size.
  int meta_size = tag->req_body_buf_.len_;
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kUserDataIndex,
         &meta_size, 4);

  uint64_t payload_size = meta_size;
  for (int i = 0; i < tag->req_tensor_count_; ++i) {
    TensorProto tensor_proto;
    if (!DataTypeCanUseMemcpy(request->feed_tensors_[i].dtype())) {
      request->feed_tensors_[i].AsProtoTensorContent(&tensor_proto);
    }
    payload_size
      += StarMessage::SerializeTensorMessage(request->feed_tensors_[i],
                                             tensor_proto, request->is_dead_[i],
                                             &tag->req_message_bufs_[i],
                                             &tag->req_tensor_bufs_[i]);
  }
  memcpy(tag->req_header_buf_.data_ + StarClientTag::kPayloadLenIndex,
         &payload_size, 8);

  ParseMetaDataCallback wrapper_parse_meta_data
    = [response, tag] (const char* buf, size_t buf_len) {
    for (uint64_t i = 0; i < buf_len; ) {
      FetchNameLenType len;
      memcpy(&len, buf + i, sizeof(FetchNameLenType));
      i += sizeof(FetchNameLenType);
      response->fetch_names_.push_back(std::string(buf + i, len));
      i += len;
    }
    auto recv_tensor_count = response->fetch_names_.size();
    CHECK_EQ(recv_tensor_count, tag->resp_tensor_count_);
    response->fetch_tensors_.resize(tag->resp_tensor_count_);

    return Status();
  };
  tag->parse_meta_data_ = std::move(wrapper_parse_meta_data);

  ParseMessageCallback wrapper_parse_message
    = [request, response, tag] (int idx, const char* tensor_msg, size_t len) {
    CHECK_EQ(StarMessage::kMessageTotalBytes, len);
    StarMessage sm;
    StarMessage::DeserializeMessage(&sm, tensor_msg);

    response->is_dead_.push_back(sm.is_dead_);
    response->data_type_.push_back(sm.data_type_);

    bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);
    if (can_memcpy) {
      // TODO(jiankeng.pt): Implement GPU device here.
      Tensor val(cpu_allocator(), sm.data_type_, sm.tensor_shape_);
      tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&val));
      tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
      tag->resp_tensor_bufs_[idx].owned_ = false;

      response->fetch_tensors_[idx] = val;
    } else {
      tag->resp_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
      tag->resp_tensor_bufs_[idx].data_ = new char[tag->resp_tensor_bufs_[idx].len_]();
    }

    return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done
    = std::bind([response, tag] (StatusCallback done,
                                 const Status& s) {
    if (s.ok()) {
      uint64_t count = tag->resp_tensor_count_;

      for (uint64_t i = 0; i < count; ++i) {
        bool can_memcpy = DataTypeCanUseMemcpy(response->data_type_[i]);
        if (can_memcpy) {
          // TODO: impl GPU device here
        } else {
          TensorProto tensor_proto;
          ParseProtoUnlimited(&tensor_proto,
                              tag->resp_tensor_bufs_[i].data_,
                              tag->resp_tensor_bufs_[i].len_);
          Tensor val;
          Status status = response->device_->MakeTensorFromProto(
              tensor_proto, AllocatorAttributes(), &val);

          if (!status.ok()) {
            LOG(ERROR) << "Failed to make tensor from proto, err msg: "
                       << status.error_message().c_str();
            done(status);
            delete tag;
            return;
          }

          response->fetch_tensors_[i] = val;
        }
      }
    } else if (s.code() != tensorflow::error::OUT_OF_RANGE) {
      LOG(ERROR) << "wrapper_done, status not ok " << s.error_message().c_str();
    }

    done(s);
    delete tag;
  },
  std::move(done),
  std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = &request->opts_;
  tag->ProcessCallOptions();
}

StarClientTag::StarClientTag(tensorflow::StarWorkerServiceMethod method,
                             WorkerEnv* env, int resp_tensor_count,
                             int req_tensor_count)
  : method_(method), status_(0), err_msg_len_(0),
    req_tensor_count_(req_tensor_count),
    resp_tensor_count_(resp_tensor_count),
    req_message_bufs_(req_tensor_count),
    req_tensor_bufs_(req_tensor_count),
    resp_tensor_bufs_(resp_tensor_count),
    parse_meta_data_(nullptr), parse_message_(nullptr), done_(nullptr),
    env_(env), call_opts_(nullptr),
    fail_fast_(false), timeout_in_ms_(0),
    resp_packet_pos_(nullptr), resp_packet_len_(0) {}

StarClientTag::~StarClientTag() {
  delete [] req_header_buf_.data_;
  delete [] req_body_buf_.data_;
  delete [] resp_body_buf_.data_;

  for (int i = 0; i < resp_tensor_count_; ++i) {
    if (resp_tensor_bufs_[i].owned_) {
      delete [] resp_tensor_bufs_[i].data_;
    }
  }

  for (uint64_t i = 0; i < req_tensor_count_; ++i) {
    delete [] req_message_bufs_[i].data_;
    if (req_tensor_bufs_[i].owned_) {
      delete [] req_tensor_bufs_[i].data_;
    }
  }
}

bool StarClientTag::IsStarRunGraph() {
  return method_ ==
      StarWorkerServiceMethod::kStarRunGraph;
}

Status StarClientTag::ParseStarRunGraphMeta(
    const char* meta, size_t len) {
  return parse_meta_data_(meta, len);
}

void StarClientTag::ProcessCallOptions() {
  if (call_opts_ != nullptr) {
    if (call_opts_->GetTimeout() > 0) {
      timeout_in_ms_ = call_opts_->GetTimeout();
    }
  }
}

bool StarClientTag::IsRecvTensor() {
  return method_ == StarWorkerServiceMethod::kRecvTensor ||
         method_ == StarWorkerServiceMethod::kFuseRecvTensor;
}

Status StarClientTag::ParseTensorMessage(
    int idx, const char* tensor_msg, size_t len) {
  return parse_message_(idx, tensor_msg, len);
}

void StarClientTag::ScheduleProcess(std::function<void()> f) {
  env_->compute_pool->Schedule(std::move(f));
}

void StarClientTag::RepeatedlyParseTensors(char* p) {
  for (auto i = 0; i < resp_tensor_count_; ++i) {
    ParseTensorMessage(i, p, StarMessage::kMessageTotalBytes);
    auto tensor_size = GetResponseTensorSize(i);
    auto tensor_buffer = GetResponseTensorBuffer(i);
    memcpy(tensor_buffer, p + StarMessage::kMessageTotalBytes, tensor_size);

    p += StarMessage::kMessageTotalBytes + tensor_size;
  }
}

Status StarClientTag::ParseResponse() {
  memcpy(&status_, resp_packet_pos_ + StarServerTag::kStatusIndex, 4);
  if (status_ != 0) {
    std::string error_msg
      = std::string(resp_packet_pos_ + StarServerTag::kHeaderSize,
                    resp_packet_len_ - StarServerTag::kHeaderSize);
    if (error_msg.empty()) {
      error_msg = "Empty error msg.";
    }

    return tensorflow::Status(static_cast<tensorflow::error::Code>(status_),
                              error_msg);
  }

  if (IsStarRunGraph()) {
    int32 meta_len = 0;
    memcpy(&meta_len, resp_packet_pos_ + StarServerTag::kUserDataIndex, 4);
    ParseStarRunGraphMeta(resp_packet_pos_ + StarServerTag::kHeaderSize,
                          meta_len);
    RepeatedlyParseTensors(resp_packet_pos_ + StarServerTag::kHeaderSize + meta_len);

  } else if (IsRecvTensor()) {
    RepeatedlyParseTensors(resp_packet_pos_ + StarServerTag::kHeaderSize);

  } else {
    memcpy(&resp_body_buf_.len_,
           resp_packet_pos_ + StarClientTag::kPayloadLenIndex, 8);
    resp_body_buf_.data_ = new char[resp_body_buf_.len_];

    memcpy(resp_body_buf_.data_,
           resp_packet_pos_ + StarServerTag::kHeaderSize,
           resp_body_buf_.len_);
  }

  return tensorflow::Status();
}

void StarClientTag::HandleResponse(Status s) {
  done_(s);
}

// payload size for non-tensor response
uint64_t StarClientTag::GetResponseBodySize() {
  return resp_body_buf_.len_;
}

// payload buffer for non-tensor response
char* StarClientTag::GetResponseBodyBuffer() {
  return resp_body_buf_.data_;
}

// tensor size
uint64_t StarClientTag::GetResponseTensorSize(int idx) {
  return resp_tensor_bufs_[idx].len_;
}

// tensor buffer
char* StarClientTag::GetResponseTensorBuffer(int idx) {
  return resp_tensor_bufs_[idx].data_;
}

} // namespace tensorflow

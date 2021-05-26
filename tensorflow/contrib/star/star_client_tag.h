#ifndef TENSORFLOW_CONTRIB_STAR_STAR_CLIENT_TAG_H_
#define TENSORFLOW_CONTRIB_STAR_STAR_CLIENT_TAG_H_

#include <functional>

#include "tensorflow/contrib/star/star_tensor_coding.h"
#include "tensorflow/contrib/star/star_worker_service_method.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/worker.pb.h"


namespace tensorflow {

typedef std::function<Status(const char*, size_t)> ParseMetaDataCallback;
typedef std::function<Status(int, const char*, size_t)> ParseMessageCallback;
typedef std::function<void(const Status&)> StatusCallback;

class StarWorkerService;
class StarTensorResponse;
class StarClientTag;
struct WorkerEnv;

void InitStarClientTag(protobuf::Message* request,
                       protobuf::Message* response,
                       StatusCallback done,
                       StarClientTag* tag,
                       CallOptions* call_opts);

void InitStarClientTag(protobuf::Message* request,
                       StarTensorResponse* response,
                       StatusCallback done,
                       StarClientTag* tag,
                       CallOptions* call_opts);

void InitStarClientTag(protobuf::Message* request,
                       StarFuseTensorResponse* response,
                       StatusCallback done,
                       StarClientTag* tag,
                       CallOptions* call_opts);

void InitStarClientTag(StarRunGraphRequest* request,
                       StarRunGraphResponse* response,
                       StatusCallback done,
                       StarClientTag* tag);

class StarClientTag {
 public:
  // Client Header 32B:
  // |  AAAA:4B | method:4B  |
  // |        tag:8B         |
  // | status:4B|user_data:4B|
  // |     payload_len:8B    |
  static const size_t kMethodIndex = 4;
  static const size_t kTagIndex = 8;
  static const size_t kStatusIndex = 16;
  static const size_t kUserDataIndex = 20;
  static const size_t kPayloadLenIndex = 24;
  static const size_t kHeaderSize = 32;

  StarClientTag(tensorflow::StarWorkerServiceMethod method,
                   WorkerEnv* env, int resp_tesnsor_count,
                   int req_tensor_count);
  virtual ~StarClientTag();

  bool IsRecvTensor();
  bool IsStarRunGraph();
  Status ParseTensorMessage(int idx, const char* tensor_msg, size_t len);
  Status ParseStarRunGraphMeta(const char* meta, size_t len);

  Status ParseResponse();
  void RepeatedlyParseTensors(char* p);
  void HandleResponse(Status s);
  void ScheduleProcess(std::function<void()> f);

  uint64_t GetResponseBodySize();
  char* GetResponseBodyBuffer();

  uint64_t GetResponseTensorSize(int idx);
  char* GetResponseTensorBuffer(int idx);

  void ProcessCallOptions();

 protected:
  StarWorkerServiceMethod method_;
  int32 status_;
  uint64_t err_msg_len_;
  int req_tensor_count_;
  int resp_tensor_count_;

  StarBuf req_header_buf_;
  StarBuf req_body_buf_;
  StarBuf resp_body_buf_;

  std::vector<StarBuf> req_message_bufs_;
  std::vector<StarBuf> req_tensor_bufs_;

  std::vector<StarBuf> resp_tensor_bufs_;

  ParseMetaDataCallback parse_meta_data_;
  ParseMessageCallback parse_message_;
  StatusCallback done_;

  WorkerEnv* env_;
  CallOptions* call_opts_;

  bool fail_fast_;
  int timeout_in_ms_;
  char* resp_packet_pos_; // Not owned.
  int64_t resp_packet_len_;

 private:
  friend void InitStarClientTag(protobuf::Message* request,
                                protobuf::Message* response,
                                StatusCallback done,
                                StarClientTag* tag,
                                CallOptions* call_opts);

  friend void InitStarClientTag(protobuf::Message* request,
                                StarTensorResponse* response,
                                StatusCallback done,
                                StarClientTag* tag,
                                CallOptions* call_opts);

  friend void InitStarClientTag(protobuf::Message* request,
                                StarFuseTensorResponse* response,
                                StatusCallback done,
                                StarClientTag* tag,
                                CallOptions* call_opts);

  friend void InitStarClientTag(StarRunGraphRequest* request,
                                StarRunGraphResponse* response,
                                StatusCallback done,
                                StarClientTag* tag);

  friend class SeastarTagFactory;
  friend class SeastarEngine;
  friend class SeastarClient;
  friend class SeastarServer;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_STAR_CLIENT_TAG_H_

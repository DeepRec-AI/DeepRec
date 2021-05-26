#ifndef TENSORFLOW_CONTRIB_STAR_STAR_SERVER_TAG_H_
#define TENSORFLOW_CONTRIB_STAR_STAR_SERVER_TAG_H_

#include <sys/time.h>
#include <functional>

#include "tensorflow/contrib/star/star_tensor_coding.h"
#include "tensorflow/contrib/star/star_worker_service_method.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/worker.pb.h"


namespace tensorflow {

typedef std::function<Status(const char*, size_t)> ParseMetaDataCallback;
typedef std::function<Status(int, const char*, size_t)> ParseMessageCallback;
typedef std::function<Status()> ParseTensorCallback;
typedef std::function<void(const Status&)> StatusCallback;

class StarWorkerService;
class StarTensorResponse;
class StarServerTag;

void InitStarServerTag(protobuf::Message* request,
                       protobuf::Message* response,
                       StarServerTag* tag);

void InitStarServerTag(protobuf::Message* request,
                       StarTensorResponse* response,
                       StarServerTag* tag,
                       StatusCallback clear);

void InitStarServerTag(protobuf::Message* request,
                       StarFuseTensorResponse* response,
                       StarServerTag* tag,
                       StatusCallback clear);

void InitStarServerTag(StarServerTag* tag);

class StarServerTag {
 public:
  // Server Header 32B:
  // |  BBBB:4B | method:4B  |
  // |        tag:8B         |
  // | status:4B|user_data:4B|
  // |     payload_len:8B    |
  static const size_t kMethodIndex = 4;
  static const size_t kTagIndex = 8;
  static const size_t kStatusIndex = 16;
  static const size_t kUserDataIndex = 20;
  static const size_t kPayloadLenIndex = 24;
  static const size_t kHeaderSize = 32;

  using HandleRequestFunction = void (StarWorkerService::*)(StarServerTag*);
  StarServerTag(StarWorkerService* star_worker_service);

  virtual ~StarServerTag();

  // Called by star engine, call the handler.
  void RecvReqDone(Status s);

  // Called by star engine.
  void SendRespDone();
  void ProcessDone(Status s);

  uint64_t GetRequestBodySize();
  char* GetRequestBodyBuffer();

  virtual void StartResp() = 0;

  void InitResponseTensorBufs(int resp_tensor_count); 
  bool IsRecvTensor();
  bool IsStarRunGraph();

  Status ParseMetaData(const char*, size_t);

  void InitRequestTensorBufs(int req_tensor_count);
  uint64_t GetRequestMessageSize(int idx);
  char* GetRequestMessageBuffer(int idx);

  uint64_t GetRequestTensorSize(int idx);
  char* GetRequestTensorBuffer(int idx);

  int GetReqTensorCount();
  Status ParseMessage(int idx, const char* tensor_msg, size_t len);
  Status ParseTensor();
  void FillRespBody();

  int64 GetRtt();
 protected:
  StarWorkerServiceMethod method_;
  uint64_t client_tag_id_;
  int32 status_;
  int req_tensor_count_;
  int resp_tensor_count_;

  StarBuf req_body_buf_;
  StarBuf resp_header_buf_;
  StarBuf resp_body_buf_;

  std::vector<StarBuf> resp_message_bufs_;
  std::vector<StarBuf> resp_tensor_bufs_;

  std::vector<StarBuf> req_tensor_bufs_;

  StarRunGraphRequest star_graph_request_;
  StarRunGraphResponse star_graph_response_;

  ParseMetaDataCallback parse_meta_data_;
  ParseMessageCallback parse_message_;
  ParseTensorCallback parse_tensor_;
  StatusCallback send_resp_;
  StatusCallback clear_;

  StarWorkerService* star_worker_service_;

  timeval start_time_ts_;
 private:
  friend void InitStarServerTag(protobuf::Message* request,
                                protobuf::Message* response,
                                StarServerTag* tag);
  friend void InitStarServerTag(protobuf::Message* request,
                                StarTensorResponse* response,
                                StarServerTag* tag,
                                StatusCallback clear);
  friend void InitStarServerTag(protobuf::Message* request,
                                StarFuseTensorResponse* response,
                                StarServerTag* tag,
                                StatusCallback clear);
  friend void InitStarServerTag(StarServerTag* tag);

  friend class StarWorkerService;
  friend class SeastarTagFactory;
  friend class SeastarEngine;
  friend class SeastarClient;
  friend class SeastarServer;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_STAR_SERVER_TAG_H_

#ifndef SERVING_PROCESSOR_SERVING_MESSAGE_CODING_H
#define SERVING_PROCESSOR_SERVING_MESSAGE_CODING_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "serving/processor/serving/predict.pb.h"

namespace tensorflow {
namespace processor {
class Request;
class Response;
class Call;
class BatchCall;
class ServingModelInfo;
class SignatureInfo;

class IParser {
 public:
  virtual Status ParseRequestFromBuf(
      const void* input_data, int input_size, Call& call,
      const SignatureInfo* info) = 0;

  virtual Status ParseRequest(
      const eas::PredictRequest& request,
      const SignatureInfo* signature_info, Call& call) = 0;

  virtual Status ParseResponseToBuf(
      const Call& call, void** output_data,
      int* output_size, const SignatureInfo* info) = 0;

  virtual Status ParseBatchRequestFromBuf(
      const void* input_data[], int* input_size,
      BatchCall& call, const SignatureInfo* info) {
    // TO be implemented
    return Status::OK();
  }

  virtual Status ParseBatchResponseToBuf(
      BatchCall& call, void* output_data[],
      int* output_size, const SignatureInfo* info) {
    // TO be implemented
    return Status::OK();
  }

  virtual Status ParseServingModelInfoToBuf(
      ServingModelInfo& model_info, void* output_data[], int* output_size) {
    // TO be implemented
    return Status::OK();
  }
};

class ProtoBufParser : public IParser {
 public:
  explicit ProtoBufParser(int thread_num);

  Status ParseRequestFromBuf(
      const void* input_data, int input_size,
      Call& call, const SignatureInfo* info) override;

  Status ParseRequest(
      const eas::PredictRequest& request,
      const SignatureInfo* signature_info, Call& call) override;

  Status ParseResponseToBuf(
      const Call& call, void** output_data,
      int* output_size, const SignatureInfo* info) override;
  
  Status ParseBatchRequestFromBuf(
      const void* input_data[], int* input_size,
      BatchCall& call, const SignatureInfo* info) override;

  Status ParseBatchResponseToBuf(
      BatchCall& call, void* output_data[],
      int* output_size, const SignatureInfo* info) override;

  Status ParseServingModelInfoToBuf(
      ServingModelInfo& model_info, void* output_data[],
      int* output_size) override;

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class FlatBufferParser : public IParser {
 public:
  explicit FlatBufferParser(int thread_num);

  Status ParseRequestFromBuf(
      const void* input_data, int input_size,
      Call& call, const SignatureInfo* info) override {
    // TO be implemented
    return Status::OK();
  }

  Status ParseRequest(
      const eas::PredictRequest& request,
      const SignatureInfo* signature_info, Call& call) override {
    return Status::OK();
  }

  Status ParseResponseToBuf(
      const Call& call, void** output_data,
      int* output_size, const SignatureInfo* info) override {
    // TO be implemented
    return Status::OK();
  }
  
  Status ParseBatchRequestFromBuf(
      const void* input_data[], int* input_size,
      BatchCall& call, const SignatureInfo* info) override {
    // TO be implemented
    return Status::OK();
  }

  Status ParseBatchResponseToBuf(
      BatchCall& call, void* output_data[],
      int* output_size, const SignatureInfo* info) override {
    // TO be implemented
    return Status::OK();
  }

  Status ParseServingModelInfoToBuf(
      ServingModelInfo& model_info, void* output_data[],
      int* output_size) override {
    // TO be implemented
    return Status::OK();
  }

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class ParserFactory {
 public:
  static IParser* GetInstance(const std::string& serialize_type,
      int thread_num) {
    if (serialize_type == "protobuf") {
      static ProtoBufParser pb_parser(thread_num);
      return &pb_parser;
    } else if (serialize_type == "flatbuffer") {
      static FlatBufferParser pb_parser(thread_num);
      return &pb_parser;
    } else {
      LOG(ERROR) << "Invalid serialize_type.";
      return nullptr;
    }
  }
};

} // processor
} // tensorflow

#endif // SERVING_PROCESSOR_SERVING_MESSAGE_CODING_H


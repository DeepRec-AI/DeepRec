#ifndef TENSORFLOW_MESSAGE_CODING_H
#define TENSORFLOW_MESSAGE_CODING_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace processor {
class Request;
class Response;
class Call;
class BatchCall;
class ServingModelInfo;
class IParser {
 public:
  virtual Status ParseRequestFromBuf(const void* input_data,
      int input_size, Call& call) = 0;

  virtual Status ParseResponseToBuf(const Call& call,
      void** output_data, int* output_size) = 0;

  virtual Status ParseBatchRequestFromBuf(const void* input_data[],
      int* input_size, BatchCall& call) {
    // TO be implemented
    return Status::OK();
  }

  virtual Status ParseBatchResponseToBuf(BatchCall& call,
      void* output_data[], int* output_size) {
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

  Status ParseRequestFromBuf(const void* input_data,
      int input_size, Call& call) override;

  Status ParseResponseToBuf(const Call& call,
      void** output_data, int* output_size) override;
  
  Status ParseBatchRequestFromBuf(const void* input_data[],
      int* input_size, BatchCall& call) override;

  Status ParseBatchResponseToBuf(BatchCall& call,
      void* output_data[], int* output_size) override;

  Status ParseServingModelInfoToBuf(
      ServingModelInfo& model_info, void* output_data[],
      int* output_size) override;

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class FlatBufferParser : public IParser {
 public:
  explicit FlatBufferParser(int thread_num);

  Status ParseRequestFromBuf(const void* input_data,
      int input_size, Call& call) override {
    // TO be implemented
    return Status::OK();
  }

  Status ParseResponseToBuf(const Call& call,
      void** output_data, int* output_size) override {
    // TO be implemented
    return Status::OK();
  }
  
  Status ParseBatchRequestFromBuf(const void* input_data[],
      int* input_size, BatchCall& call) override {
    // TO be implemented
    return Status::OK();
  }

  Status ParseBatchResponseToBuf(BatchCall& call,
      void* output_data[], int* output_size) override {
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

#endif // TENSORFLOW_MESSAGE_CODING_H


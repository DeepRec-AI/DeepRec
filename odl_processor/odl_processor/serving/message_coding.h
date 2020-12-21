#ifndef TENSORFLOW_MESSAGE_CODING_H
#define TENSORFLOW_MESSAGE_CODING_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {
class Request;
class Response;
class IParser {
 public:
  virtual Status ParseRequestFromBuf(const void* input_data,
      int input_size, Request& req) = 0;

  virtual Status ParseResponseToBuf(const Request& req,
      const Response& resp, void** output_data, int* output_size) = 0;
};

class ProtoBufParser : public IParser {
 public:
  Status ParseRequestFromBuf(const void* input_data,
      int input_size, Request& req) override;

  Status ParseResponseToBuf(const Request& req, const Response& resp,
      void** output_data, int* output_size) override;
};

class FlatBufferParser : public IParser {
 public:
  Status ParseRequestFromBuf(const void* input_data,
      int input_size, Request& req) override;

  Status ParseResponseToBuf(const Request& req, const Response& resp,
      void** output_data, int* output_size) override;
};

class ParserFactory {
 public:
  static IParser* GetInstance(const std::string& serialize_type) {
    if (serialize_type == "protobuf") {
      static ProtoBufParser pb_parser;
      return &pb_parser;
    } else if (serialize_type == "flatbuffer") {
      static FlatBufferParser pb_parser;
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


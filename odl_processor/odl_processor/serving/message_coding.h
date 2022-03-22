#ifndef TENSORFLOW_MESSAGE_CODING_H
#define TENSORFLOW_MESSAGE_CODING_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace processor {
class Request;
class Response;
class Parser {
 public:
  Status ParseRequestFromProto(const void* input_data, int input_size,
      Request& req);

  Status ParseResponseToProto(const Request& req, const Response& resp,
      void** output_data, int* output_size); 
};

} // processor
} // tensorflow

#endif // TENSORFLOW_MESSAGE_CODING_H


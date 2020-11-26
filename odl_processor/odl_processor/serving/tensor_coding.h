#ifndef TENSORFLOW_TENSOR_CODING_H
#define TENSORFLOW_TENSOR_CODING_H

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class Tensor;
class TensorInfo;

namespace processor {
class Request {
};

class Response {
};

template<typename Req, typename Resp>
struct Call {
  Req request;
  Resp response;
};
typedef Call<Request, Response> TensorCall;

class RequestFactory {
};

class ResponseFactory {
};

} // processor
} // tensorflow

#endif // TENSORFLOW_TENSOR_CODING_H


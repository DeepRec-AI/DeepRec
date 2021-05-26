#ifndef TENSORFLOW_CONTRIB_STAR_STAR_MESSAGE_H_
#define TENSORFLOW_CONTRIB_STAR_STAR_MESSAGE_H_

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/contrib/star/star_tensor_coding.h"


namespace tensorflow {
static const int _8KB = 8 * 1024;

// message for recv tensor response
struct StarMessage {
  bool is_dead_;
  DataType data_type_;
  TensorShape tensor_shape_;
  uint64_t tensor_bytes_;

  // |is_dead|...
  // |    1B |...
  // ...|data_type|tensor_shape|tensor_bytes|tensor_buffer
  // ...|   XB    |    XB      |    8B      |...

  static const size_t kIsDeadStartIndex = 0;
  static const size_t kDataTypeStartIndex =
      kIsDeadStartIndex + sizeof(is_dead_);
  static const size_t kTensorShapeStartIndex =
      kDataTypeStartIndex + sizeof(data_type_);
  static const size_t kTensorBytesStartIndex =
      kTensorShapeStartIndex + sizeof(TensorShape);
  static const size_t kTensorBufferStartIndex =
      kTensorBytesStartIndex + sizeof(tensor_bytes_);
  static const size_t kMessageTotalBytes = kTensorBufferStartIndex;
  static const size_t kStarMessageBufferSize = kMessageTotalBytes;
  static void SerializeMessage(const StarMessage& rm, char* data);
  static void DeserializeMessage(StarMessage* rm, const char* data);
  static uint64_t SerializeTensorMessage(
      const Tensor& in, const TensorProto& inp,
      bool is_dead, StarBuf* message_buf,
      StarBuf* tensor_buf);
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_STAR_MESSAGE_H_

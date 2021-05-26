#include "tensorflow/contrib/star/star_message.h"


namespace tensorflow {

void StarMessage::DeserializeMessage(StarMessage* sm, const char* message) {
  // data_type, tensor_bytes, tensor_shape, is_dead
  memcpy(&sm->is_dead_, &message[kIsDeadStartIndex], sizeof(sm->is_dead_));
  memcpy(&sm->data_type_, &message[kDataTypeStartIndex],
         sizeof(sm->data_type_));
  memcpy(&sm->tensor_shape_, &message[kTensorShapeStartIndex],
         sizeof(sm->tensor_shape_));
  memcpy(&sm->tensor_bytes_, &message[kTensorBytesStartIndex],
         sizeof(sm->tensor_bytes_));
}

void StarMessage::SerializeMessage(const StarMessage& sm, char* message) {
  // is_dead, data_type, tensor_shape, tensor_bytes
  memcpy(&message[kIsDeadStartIndex], &sm.is_dead_, sizeof(sm.is_dead_));

  memcpy(&message[kDataTypeStartIndex], &sm.data_type_,
         sizeof(sm.data_type_));
  memcpy(&message[kTensorShapeStartIndex], &sm.tensor_shape_,
         sizeof(sm.tensor_shape_));
  memcpy(&message[kTensorBytesStartIndex], &sm.tensor_bytes_,
           sizeof(sm.tensor_bytes_));
}

uint64_t StarMessage::SerializeTensorMessage(
    const Tensor& in, const TensorProto& inp,
    bool is_dead, StarBuf* message_buf,
    StarBuf* tensor_buf) {
  StarMessage sm;
  sm.tensor_shape_ = in.shape();
  sm.data_type_ = in.dtype();
  sm.is_dead_ = is_dead;

  bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

  if (can_memcpy) {
    sm.tensor_bytes_ = in.TotalBytes();

    tensor_buf->len_ = sm.tensor_bytes_;
    tensor_buf->data_ = const_cast<char*>(in.tensor_data().data());
    tensor_buf->owned_ = false;
  } else {
    sm.tensor_bytes_ = inp.ByteSize();

    tensor_buf->len_ = sm.tensor_bytes_;
    tensor_buf->data_ = new char[tensor_buf->len_]();
    inp.SerializeToArray(tensor_buf->data_, tensor_buf->len_);
  }

  message_buf->len_ = StarMessage::kMessageTotalBytes;
  message_buf->data_ = new char[message_buf->len_];
  StarMessage::SerializeMessage(sm, message_buf->data_);

  return StarMessage::kMessageTotalBytes + sm.tensor_bytes_;
}

} // namespace tensorflow

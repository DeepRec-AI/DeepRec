#include "tensorflow/contrib/star/star_tensor_coding.h"
#include "tensorflow/core/common_runtime/device.h"


namespace tensorflow {

void StarTensorResponse::InitAlloc(
    Device* d, const AllocatorAttributes& aa) {
  Clear();
  device_ = d;
  alloc_attrs_ = aa;
  const DeviceAttributes& da = d->attributes();
  if (alloc_attrs_.on_host() || da.device_type() == "CPU") {
    on_host_ = true;
  }
  allocator_ = device_->GetAllocator(alloc_attrs_);
}

void StarTensorResponse::Clear() {
  on_host_ = false;
  device_ = nullptr;
  alloc_attrs_ = AllocatorAttributes();
  allocator_ = nullptr;
  tensor_ = Tensor();
  tensor_proto_ = TensorProto();
}

void StarFuseTensorResponse::Clear() {
  StarTensorResponse::Clear();
  fuse_count_ = 0;
  tensors_.clear();
  tensor_protos_.clear();
  is_deads_.clear();
}

void StarRunGraphRequest::EncodeRequest(StarBuf* star_buf) {
  star_buf->len_ = 0;
  star_buf->len_ += sizeof(CounterType);
  star_buf->len_ += feed_names_.size() * sizeof(FeedNameLenType);
  for (uint64_t i = 0; i < feed_names_.size(); i++) {
    star_buf->len_ += feed_names_[i].length();
  }
  star_buf->len_ += sizeof(CounterType);
  star_buf->len_ += fetch_names_.size() * sizeof(FetchNameLenType);
  for (uint64_t i = 0; i < fetch_names_.size(); i++) {
    star_buf->len_ += fetch_names_[i].length();
  }
  star_buf->len_ += sizeof(GraphHandleLenType);
  star_buf->len_ += graph_handle_.length();
  star_buf->len_ += sizeof(StepIdType);
  star_buf->len_ += sizeof(int32_t);
  star_buf->len_ += sizeof(bool);
  star_buf->len_ += 2 * sizeof(GraphHandleLenType);
  star_buf->len_ += op_span_context_.length();
  star_buf->len_ += op_device_name_.length();

  star_buf->data_ = new char[star_buf->len_]();

  uint64_t offset = 0;
  CounterType vec_size = feed_names_.size();
  uint16_t len = 0;
  memcpy(star_buf->data_, &vec_size, sizeof(CounterType));
  offset += sizeof(CounterType);
  for (uint64_t i = 0; i < feed_names_.size(); i++) {
    len = feed_names_[i].length();
    memcpy(star_buf->data_ + offset, &len, sizeof(FeedNameLenType));
    offset += sizeof(FeedNameLenType);
    memcpy(star_buf->data_ + offset, feed_names_[i].c_str(), len);
    offset += len;
  }

  vec_size = fetch_names_.size();
  memcpy(star_buf->data_ + offset, &vec_size, sizeof(CounterType));
  offset += sizeof(CounterType);
  for (uint64_t i = 0; i < fetch_names_.size(); i++) {
    len = fetch_names_[i].length();
    memcpy(star_buf->data_ + offset, &len, sizeof(FetchNameLenType));
    offset += sizeof(FetchNameLenType);
    memcpy(star_buf->data_ + offset, fetch_names_[i].c_str(), len);
    offset += len;
  }

  len = graph_handle_.length();
  memcpy(star_buf->data_ + offset, &len, sizeof(GraphHandleLenType));
  offset += sizeof(GraphHandleLenType);
  memcpy(star_buf->data_ + offset, graph_handle_.c_str(), len);
  offset += len;
  memcpy(star_buf->data_ + offset, &step_id_, sizeof(StepIdType));
  offset += sizeof(StepIdType);
  memcpy(star_buf->data_ + offset, &ps_graph_count_, sizeof(int32_t));
  offset += sizeof(int32_t);
  memcpy(star_buf->data_ + offset, &should_tracing_, sizeof(bool));
  offset += sizeof(bool);
  len = op_span_context_.length();
  memcpy(star_buf->data_ + offset, &len, sizeof(GraphHandleLenType));
  offset += sizeof(GraphHandleLenType);
  memcpy(star_buf->data_ + offset, op_span_context_.c_str(), len);
  offset += len;
  len = op_device_name_.length();
  memcpy(star_buf->data_ + offset, &len, sizeof(GraphHandleLenType));
  offset += sizeof(GraphHandleLenType);
  memcpy(star_buf->data_ + offset, op_device_name_.c_str(), len);
}

void StarRunGraphRequest::DecodeRequest(const char* buf, size_t len) {
  CounterType feed_count = 0;
  CounterType fetch_count = 0;
  uint64_t offset = 0;
  FeedNameLenType feed_len = 0;
  FetchNameLenType fetch_len = 0;

  memcpy(&feed_count, buf, sizeof(CounterType));
  offset += sizeof(CounterType);
  feed_names_.resize(feed_count);
  feed_tensors_.resize(feed_count);
  for (int i = 0; i < feed_count; ++i) {
    memcpy(&feed_len, buf + offset, sizeof(FeedNameLenType));
    offset += sizeof(FeedNameLenType);
    feed_names_[i] = std::string(buf + offset, feed_len);
    offset += feed_len;
  }

  memcpy(&fetch_count, buf + offset, sizeof(CounterType));
  offset += sizeof(CounterType);
  fetch_names_.resize(fetch_count);
  for (int i = 0; i < fetch_count; ++i) {
    memcpy(&fetch_len, buf + offset, sizeof(FetchNameLenType));
    offset += sizeof(FetchNameLenType);
    fetch_names_[i] = std::string(buf + offset, fetch_len);
    offset += fetch_len;
  }

  GraphHandleLenType graph_handle_len = 0;
  memcpy(&graph_handle_len, buf + offset, sizeof(GraphHandleLenType));
  offset += sizeof(GraphHandleLenType);
  graph_handle_ = std::string(buf + offset, graph_handle_len);
  offset += graph_handle_len;
  memcpy(&step_id_, buf + offset, sizeof(StepIdType));
  offset += sizeof(StepIdType);
  memcpy(&ps_graph_count_, buf + offset, sizeof(int32_t));
  offset += sizeof(int32_t);
  memcpy(&should_tracing_, buf + offset, sizeof(bool));
  offset += sizeof(bool);
  GraphHandleLenType span_ctx_len = 0;
  memcpy(&span_ctx_len, buf + offset, sizeof(GraphHandleLenType));
  offset += sizeof(GraphHandleLenType);
  op_span_context_ = std::string(buf + offset, span_ctx_len);
  offset += span_ctx_len;
  GraphHandleLenType dev_name_len = 0;
  memcpy(&dev_name_len, buf + offset, sizeof(GraphHandleLenType));
  offset += sizeof(GraphHandleLenType);
  op_device_name_ = std::string(buf + offset, dev_name_len);
  offset += dev_name_len;

  if (offset != len) {
    LOG(FATAL) << "Error meta data length!";
  }
}

} // namespace tensorflow

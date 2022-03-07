#include "odl_processor/serving/message_coding.h"
#include "odl_processor/serving/util.h"

namespace tensorflow {
namespace processor {
ProtoBufParser::ProtoBufParser(int thread_num) {
  thread_pool_.reset(new thread::ThreadPool(Env::Default(), "",
      thread_num));
}

Status ProtoBufParser::ParseRequestFromBuf(const void* input_data,
    int input_size, Call& call) {
  eas::PredictRequest request;
  request.ParseFromArray(input_data, input_size);

  for (auto& input : request.inputs()) {
    call.request.inputs.emplace_back(input.first, util::Proto2Tensor(input.second));
  }

  call.request.output_tensor_names =
      std::vector<std::string>(request.output_filter().begin(),
                               request.output_filter().end());

  return Status::OK();
}

Status ProtoBufParser::ParseResponseToBuf(const Call& call,
    void** output_data, int* output_size) {
  eas::PredictResponse response = util::Tensor2Response(call.request,
      call.response);
  *output_size = response.ByteSize();
  *output_data = new char[*output_size];
  response.SerializeToArray(*output_data, *output_size);
  return Status::OK();
}

Status ProtoBufParser::ParseBatchRequestFromBuf(
    const void* input_data[], int* input_size, BatchCall& call) {
  auto size = sizeof(input_data) / sizeof(void*);
  call.call_num = size;
  auto do_work = [&call, input_data, input_size](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      eas::PredictRequest request;
      request.ParseFromArray(input_data[i], input_size[i]);
      
      for (auto& input : request.inputs()) {
        call.request[i].inputs.emplace_back(input.first, util::Proto2Tensor(input.second));
      } 
 
      if (i == 0) { 
        call.request[0].output_tensor_names =
            std::vector<std::string>(request.output_filter().begin(),
                                     request.output_filter().end());
      }
    }
  };
  thread_pool_->ParallelFor(size, 10000, do_work);

  return call.BatchRequest();
}

Status ProtoBufParser::ParseBatchResponseToBuf(
    BatchCall& call, void* output_data[], int* output_size) {
  //TF_RETURN_IF_ERROR(call.SplitResponse());
  call.SplitResponse();
  auto do_work = [&call, output_data, output_size](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      eas::PredictResponse response = util::Tensor2Response(call.request[i],
          call.response[i]);
      output_size[i] = response.ByteSize();
      output_data[i] = new char[*output_size];
      response.SerializeToArray(output_data[i], output_size[i]); 
    }
  };
  thread_pool_->ParallelFor(call.call_num, 10000, do_work);
  return Status::OK();
}

Status ProtoBufParser::ParseServingModelInfoToBuf(
    ServingModelInfo& model_info, void* output_data[],
    int* output_size) {
  eas::ServingModelInfo info;
  *info.mutable_model_path() = model_info.model_path;
  *output_size = info.ByteSize();
  *output_data = new char[*output_size];
  info.SerializeToArray(*output_data, *output_size);
  return Status::OK();
}

FlatBufferParser::FlatBufferParser(int thread_num) {
  thread_pool_.reset(new thread::ThreadPool(Env::Default(), "",
      thread_num));
}

} // processor
} // tensorflow

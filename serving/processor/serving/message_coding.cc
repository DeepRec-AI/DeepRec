#include "serving/processor/serving/model_message.h"
#include "serving/processor/serving/message_coding.h"
#include "serving/processor/serving/util.h"

namespace tensorflow {
namespace processor {
ProtoBufParser::ProtoBufParser(int thread_num) {
  thread_pool_.reset(new thread::ThreadPool(Env::Default(), "",
      thread_num));
}

Status ProtoBufParser::ParseRequest(
    const eas::PredictRequest& request,
    const SignatureInfo* signature_info, Call& call) {
  for (auto& input : request.inputs()) {
    if (signature_info->input_key_idx.find(input.first) ==
        signature_info->input_key_idx.end()) {
      LOG(FATAL) << "Request contain invalid input key : " << input.first;
    }
    int idx = signature_info->input_key_idx.at(input.first);
    auto pb_to_tensor = util::Proto2Tensor(input.first, input.second);
    if (!pb_to_tensor.status.ok()) {
      return pb_to_tensor.status;
    }
    call.request.inputs.emplace_back(
        signature_info->input_value_name[idx],
        std::move(pb_to_tensor.tensor));
  }

  if (request.output_filter().size() > 0) {
    call.request.output_tensor_names.reserve(request.output_filter().size());
    for (auto key : request.output_filter()) {
      if (signature_info->output_key_idx.find(key) ==
          signature_info->output_key_idx.end()) {
        LOG(FATAL) << "Request contain invalid output filter: " << key;
      }
      call.request.output_tensor_names.emplace_back(
          signature_info->output_value_name[signature_info->output_key_idx.at(key)]);
    }
  } else {
    call.request.output_tensor_names =
        signature_info->output_value_name;
  }

  return Status::OK();
}

Status ProtoBufParser::ParseRequestFromBuf(
    const void* input_data, int input_size, Call& call,
    const SignatureInfo* signature_info) {
  eas::PredictRequest request;
  bool success = request.ParseFromArray(input_data, input_size);
  if (!success) {
    LOG(ERROR) << "Parse request from array failed, input_data: " << input_data
               << ", input_size: " << input_size;
    return Status(errors::Code::INVALID_ARGUMENT, "Please check the input data.");
  }

  return ParseRequest(request, signature_info, call);
}

Status ProtoBufParser::ParseResponseToBuf(
    const Call& call, void** output_data, int* output_size,
    const SignatureInfo* signature_info) {
  eas::PredictResponse response = util::Tensor2Response(call.request,
      call.response, signature_info);
  *output_size = response.ByteSize();
  *output_data = new char[*output_size];
  bool success = response.SerializeToArray(*output_data, *output_size);
  if (!success) {
    LOG(ERROR) << "Parse reponse to array failed. " << response.DebugString();
    return Status(errors::Code::INTERNAL, "Serialize response to array failed.");
  }
  return Status::OK();
}

Status ProtoBufParser::ParseBatchRequestFromBuf(
    const void* input_data[], int* input_size,
    BatchCall& call, const SignatureInfo* signature_info) {
  auto size = sizeof(input_data) / sizeof(void*);
  call.call_num = size;
  auto do_work = [&call, input_data, input_size,
                  signature_info](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      eas::PredictRequest request;
      request.ParseFromArray(input_data[i], input_size[i]);
      
      for (auto& input : request.inputs()) {
        if (signature_info->input_key_idx.find(input.first) ==
            signature_info->input_key_idx.end()) {
          LOG(FATAL) << "Request contain invalid input key : " << input.first;
        }
        int idx = signature_info->input_key_idx.at(input.first);
        auto pb_to_tensor = util::Proto2Tensor(input.first, input.second);
        if (!pb_to_tensor.status.ok()) {
          return pb_to_tensor.status;
        }
        call.request[i].inputs.emplace_back(
            signature_info->input_value_name[idx],
            std::move(pb_to_tensor.tensor));
      }
 
      if (i == 0) {
        if (request.output_filter().size() > 0) {
          call.request[0].output_tensor_names.reserve(request.output_filter().size());
          for (auto key : request.output_filter()) {
            if (signature_info->output_key_idx.find(key) ==
                signature_info->output_key_idx.end()) {
              LOG(FATAL) << "Request contain invalid output filter: " << key;
            }
            call.request[0].output_tensor_names.emplace_back(
                signature_info->output_value_name[signature_info->output_key_idx.at(key)]);
          }
        } else {
          call.request[0].output_tensor_names =
              signature_info->output_value_name;
        }
      }
    }
  };
  thread_pool_->ParallelFor(size, 10000, do_work);

  return call.BatchRequest();
}

Status ProtoBufParser::ParseBatchResponseToBuf(
    BatchCall& call, void* output_data[],
    int* output_size, const SignatureInfo* signature_info) {
  //TF_RETURN_IF_ERROR(call.SplitResponse());
  call.SplitResponse();
  auto do_work = [&call, output_data, output_size,
	              signature_info](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      eas::PredictResponse response = util::Tensor2Response(call.request[i],
          call.response[i], signature_info);
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

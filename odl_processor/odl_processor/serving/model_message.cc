#include "odl_processor/serving/model_message.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
namespace processor {
namespace {
Status ValidateShape(const std::vector<Request>& requests) {
  for (int i = 0; i < requests[0].inputs.size(); ++i) {
    auto shape = requests[0].inputs[i].second.shape();
    for(auto it : requests) {
      if (shape != it.inputs[i].second.shape()) {
        return Status(error::Code::INVALID_ARGUMENT,
            "Invalid input shapes when batched process.");
      }
    }
  }
  return Status::OK();
}

template<typename T>
Tensor BatchOneInput(const std::vector<Request>& requests, int index) {
  TensorShape tensor_shape;
  tensor_shape.AddDim(requests.size());
 
  size_t single_input_size = 1;
  auto input = requests[0].inputs[index].second;
  for (int j = 0; j < input.dims(); ++j) {
    tensor_shape.AddDim(input.dim_size(j));
    single_input_size *= input.dim_size(j);
  }
  
  Tensor batched_tensor(input.dtype(), tensor_shape);
  auto batched_tensor_flat = batched_tensor.flat<T>();
  size_t current_pos = 0;
  for (auto req : requests) {
    auto flat = req.inputs[index].second.flat<T>();
    memcpy(batched_tensor_flat.data() + current_pos,
           flat.data(), single_input_size);
    current_pos += single_input_size;
  }

  return batched_tensor;
}

template<typename T>
void SplitOneOutput(Response& batched_response,
    int index, int call_num, std::vector<Tensor>& ret) {
  Tensor& batched_tensor = batched_response.outputs[index];
  auto batched_tensor_shape = batched_tensor.shape();
  auto batched_flat = batched_tensor.flat<T>();
  size_t current_pos = 0;

  size_t single_output_size = 1;
  TensorShape tensor_shape;
  for (int j = 1; j < batched_tensor_shape.dims(); ++j) {
    tensor_shape.AddDim(batched_tensor_shape.dim_size(j));
    single_output_size *= batched_tensor_shape.dim_size(j);
  }
  for (int i = 0; i < call_num; ++i) {
    Tensor t(batched_tensor.dtype(), tensor_shape);
    auto flat = t.flat<T>();

    memcpy(flat.data(), batched_flat.data() + current_pos,
        single_output_size);
    current_pos += single_output_size;
    ret[i] = t;
  }
}
}

Status BatchCall::BatchRequest() {
  TF_RETURN_IF_ERROR(ValidateShape(request));
  for (int i = 0; i < request[0].inputs.size(); ++i) {
    auto& t = request[0].inputs[i].second;
    switch (t.dtype()) {
      case DT_FLOAT: {
        auto batched_tensor = BatchOneInput<float>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_DOUBLE: {
        auto batched_tensor = BatchOneInput<double>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_INT32: {
        auto batched_tensor = BatchOneInput<int32_t>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_UINT8: {
        auto batched_tensor = BatchOneInput<uint8>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_INT16: {
        auto batched_tensor = BatchOneInput<int16>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_UINT16: {
        auto batched_tensor = BatchOneInput<uint16>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_INT8: {
        auto batched_tensor = BatchOneInput<int8>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_COMPLEX64: {
        auto batched_tensor = BatchOneInput<complex64>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_COMPLEX128: {
        auto batched_tensor = BatchOneInput<complex128>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_INT64: {
        auto batched_tensor = BatchOneInput<int64>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_BOOL: {
        auto batched_tensor = BatchOneInput<bool>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_QINT8: {
        auto batched_tensor = BatchOneInput<qint8>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_QUINT8: {
        auto batched_tensor = BatchOneInput<quint8>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_QINT32: {
        auto batched_tensor = BatchOneInput<qint32>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_QINT16: {
        auto batched_tensor = BatchOneInput<qint16>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_QUINT16: {
        auto batched_tensor = BatchOneInput<quint16>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_BFLOAT16: {
        auto batched_tensor = BatchOneInput<bfloat16>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      case DT_HALF: {
        auto batched_tensor = BatchOneInput<Eigen::half>(request, i);
        batched_request.inputs.emplace_back(request[0].inputs[i].first,
            batched_tensor);
        break;
      }
      default: {
        LOG(FATAL) << "Invalid Input.";
        break;
      }
    }
  }

  batched_request.output_tensor_names = request[0].output_tensor_names;
  return Status::OK();
}

Status BatchCall::SplitResponse() {
  response.resize(call_num);
  for (int i = 0; i < batched_response.outputs.size(); ++i) {
     std::vector<Tensor> splited_vec(call_num);
     auto& t = batched_response.outputs[i];
     switch (t.dtype()) {
       case DT_FLOAT: {
         SplitOneOutput<float>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_DOUBLE: {
         SplitOneOutput<double>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_INT32: {
         SplitOneOutput<int>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_UINT8: {
         SplitOneOutput<uint8>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_INT16: {
         SplitOneOutput<int16>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_UINT16: {
         SplitOneOutput<uint16>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_INT8: {
         SplitOneOutput<int8>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_COMPLEX64: {
         SplitOneOutput<complex64>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_COMPLEX128: {
         SplitOneOutput<complex128>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_INT64: {
         SplitOneOutput<int64>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_BOOL: {
         SplitOneOutput<bool>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_QINT8: {
         SplitOneOutput<qint8>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_QUINT8: {
         SplitOneOutput<quint8>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_QINT32: {
         SplitOneOutput<qint32>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_QINT16: {
         SplitOneOutput<qint16>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_QUINT16: {
         SplitOneOutput<quint16>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_BFLOAT16: {
         SplitOneOutput<bfloat16>(batched_response, i, call_num, splited_vec);
         break;
       }
       case DT_HALF: {
         SplitOneOutput<Eigen::half>(batched_response, i, call_num, splited_vec);
         break;
       }
       default: {
         LOG(FATAL) << "Output tensor invalid type.";
       }
     }

     if (splited_vec.size() != call_num) {
       return Status(error::Code::INVALID_ARGUMENT,
           "Invalid output tensor.");
     }
     for (auto j = 0; j < call_num; ++j) {
       response[j].outputs[i] = splited_vec[j];
     }
  }
  return Status::OK();
}

} // processor
} // tensorflow

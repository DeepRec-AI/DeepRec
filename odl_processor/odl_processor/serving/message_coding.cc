#include "odl_processor/serving/message_coding.h"
#include "odl_processor/serving/model_message.h"
#include "odl_processor/serving/tf_predict.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
namespace {
Tensor Proto2Tensor(const eas::ArrayProto& input) {
  TensorShape tensor_shape;
  int64 total_size = 1;
  for (int i = 0; i < input.array_shape().dim_size(); ++i) {
    tensor_shape.AddDim(input.array_shape().dim(i));
    total_size *= input.array_shape().dim(i);
  }

  switch (input.dtype()) {
    case tensorflow::eas::DT_FLOAT: {
      if (total_size != input.float_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_FLOAT, tensor_shape);
      auto flat = tensor.flat<float>();
      memcpy(flat.data(), input.float_val().data(),
          input.float_val_size() * sizeof(float));

      return tensor;
    }
    case tensorflow::eas::DT_DOUBLE: {
      if (total_size != input.double_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_DOUBLE, tensor_shape);
      auto flat = tensor.flat<double>();
      memcpy(flat.data(), input.double_val().data(),
          input.double_val_size() * sizeof(double));
      return tensor;
    }
    case tensorflow::eas::DT_INT32: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_INT32, tensor_shape);
      auto flat = tensor.flat<int>();
      memcpy(flat.data(), input.int_val().data(),
          input.int_val_size() * sizeof(int));
      return tensor;
    }
    case tensorflow::eas::DT_UINT8: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(tensorflow::DT_UINT8, tensor_shape);
      auto flat = tensor.flat<uint8>();
      for (int i = 0; i < input.int_val_size(); i++) {
        flat(i) = (uint8)input.int_val(i);
      }
      return tensor;
    }
    case tensorflow::eas::DT_INT16: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(tensorflow::DT_INT16, tensor_shape);
      auto flat = tensor.flat<int16>();
      for (int i = 0; i < input.int_val_size(); i++) {
        flat(i) = (int16)input.int_val(i);
      }
      return tensor;
    }
    case tensorflow::eas::DT_UINT16: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(tensorflow::DT_UINT16, tensor_shape);
      auto flat = tensor.flat<uint16>();
      for (int i = 0; i < input.int_val_size(); i++) {
        flat(i) = (uint16)input.int_val(i);
      }
      return tensor;
    }
    case tensorflow::eas::DT_INT8: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(tensorflow::DT_INT8, tensor_shape);
      auto flat = tensor.flat<int8>();
      for (int i = 0; i < input.int_val_size(); i++) {
        flat(i) = (int8)input.int_val(i);
      }
      return tensor;
    }
    case tensorflow::eas::DT_STRING: {
      if (total_size != input.string_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(tensorflow::DT_STRING, tensor_shape);
      auto flat = tensor.flat<std::string>();
      for (int i = 0; i < input.string_val_size(); i++) {
        flat(i) = input.string_val(i);
      }
      return tensor;
    }
    case tensorflow::eas::DT_COMPLEX64: {
      if (total_size != input.float_val_size() / 2) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(tensorflow::DT_COMPLEX64, tensor_shape);
      auto flat = tensor.flat<complex64>();
      for (int i = 0; i < input.float_val_size(); i += 2) {
        flat(i) = complex64(input.float_val(i), input.float_val(i + 1));
      }
      return tensor;
    }
    case tensorflow::eas::DT_COMPLEX128: {
      if (total_size != input.double_val_size() / 2) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(tensorflow::DT_COMPLEX128, tensor_shape);
      auto flat = tensor.flat<complex128>();
      for (int i = 0; i < input.double_val_size(); i += 2) {
        flat(i) = complex64(input.double_val(i), input.double_val(i + 1));
      }
      return tensor;
    }
    case tensorflow::eas::DT_INT64: {
      if (total_size != input.int64_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_INT64, tensor_shape);
      auto flat = tensor.flat<int64>();
      memcpy(flat.data(), input.int64_val().data(),
          input.int64_val_size() * sizeof(int64));
      return tensor;
    }
    case tensorflow::eas::DT_BOOL: {
      if (total_size != input.bool_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_BOOL, tensor_shape);
      auto flat = tensor.flat<bool>();
      for (int i = 0; i < input.bool_val_size(); ++i) {
        flat(i) = input.bool_val(i);
      }
      return tensor;
    }
    case tensorflow::eas::DT_QINT8: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_QINT8, tensor_shape);
      auto flat = tensor.flat<qint8>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = qint8(input.int_val(i));
      }
      return tensor;
    }
    case tensorflow::eas::DT_QUINT8: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_QUINT8, tensor_shape);
      auto flat = tensor.flat<quint8>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = quint8(input.int_val(i));
      }
      return tensor;
    }
    case tensorflow::eas::DT_QINT32: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_QINT32, tensor_shape);
      auto flat = tensor.flat<qint32>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = qint32(input.int_val(i));
      }
      return tensor;
    }
    case tensorflow::eas::DT_QINT16: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_QINT16, tensor_shape);
      auto flat = tensor.flat<qint16>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = qint16(input.int_val(i));
      }
      return tensor;
    }
    case tensorflow::eas::DT_QUINT16: {
      if (total_size != input.int_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_QUINT16, tensor_shape);
      auto flat = tensor.flat<quint16>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = quint16(input.int_val(i));
      }
      return tensor;
    }
    case tensorflow::eas::DT_BFLOAT16: {
      if (total_size != input.float_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_BFLOAT16, tensor_shape);
      auto flat = tensor.flat<bfloat16>();
      tensorflow::FloatToBFloat16(input.float_val().data(),
                                  flat.data(),
                                  input.float_val_size());
      return tensor;
    }
    case tensorflow::eas::DT_HALF: {
      if (total_size != input.float_val_size()) {
        // TODO: should skip current request
        LOG(FATAL) << "Invalid input.";
      }
      Tensor tensor(DT_HALF, tensor_shape);
      auto flat = tensor.flat<Eigen::half>();
      for (int i = 0; i < input.float_val_size(); ++i) {
        flat(i) = Eigen::half(input.float_val(i));
      }
      return tensor;
    }
    case tensorflow::eas::DT_RESOURCE: {
      LOG(FATAL) << "Input Tensor Not Support this DataType: DT_RESOURCE";
      break;
    }
    case tensorflow::eas::DT_VARIANT: {
      LOG(FATAL) << "Input Tensor Not Support this DataType: DT_VARIANT";
      break;
    }
    default: {
      LOG(FATAL) << "Input Tensor Not Support this DataType";
      break;
    }
  }
  return Tensor();
}

eas::PredictResponse Tensor2Response(const processor::Request& req,
    const processor::Response& resp) {
  eas::PredictResponse response;
  const auto& output_tensor_names = req.output_tensor_names;
  const auto & outputs = resp.outputs;

  for (size_t i = 0; i < outputs.size(); ++i) {
    eas::ArrayProto output;
    int64 total_dim_size = 1;
    for (int j = 0; j < outputs[i].dims(); ++j) {
      int64 dim_size = outputs[i].dim_size(j);
      output.mutable_array_shape()->add_dim(dim_size);
      total_dim_size *= dim_size;
    }

    switch (outputs[i].dtype()) {
      case DT_FLOAT: {
        output.set_dtype(eas::DT_FLOAT);
        auto flat = outputs[i].flat<float>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_float_val(flat(j));
        }
        break;
      }
      case DT_DOUBLE: {
        output.set_dtype(eas::DT_DOUBLE);
        auto flat = outputs[i].flat<double>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_double_val(flat(j));
        }
        break;
      }
      case DT_INT32: {
        output.set_dtype(eas::DT_INT32);
        auto flat = outputs[i].flat<int>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j));
        }
        break;
      }
      case DT_UINT8: {
        output.set_dtype(eas::DT_UINT8);
        auto flat = outputs[i].flat<uint8>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val((int)flat(j));
        }
        break;
      }
      case DT_INT16: {
        output.set_dtype(eas::DT_INT16);
        auto flat = outputs[i].flat<int16>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val((int)flat(j));
        }
        break;
      }
      case DT_INT8: {
        output.set_dtype(eas::DT_INT8);
        auto flat = outputs[i].flat<int8>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val((int)flat(j));
        }
        break;
      }
      case DT_QINT8: {
        output.set_dtype(eas::DT_QINT8);
        auto flat = outputs[i].flat<qint8>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_QUINT8: {
        output.set_dtype(eas::DT_QUINT8);
        auto flat = outputs[i].flat<quint8>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_QINT32: {
        output.set_dtype(eas::DT_QINT32);
        auto flat = outputs[i].flat<qint32>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_QINT16: {
        output.set_dtype(eas::DT_QINT16);
        auto flat = outputs[i].flat<qint16>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_QUINT16: {
        output.set_dtype(eas::DT_QUINT16);
        auto flat = outputs[i].flat<quint16>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_UINT16: {
        output.set_dtype(eas::DT_UINT16);
        auto flat = outputs[i].flat<uint16>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val((int)flat(j));
        }
        break;
      }
      case DT_INT64: {
        output.set_dtype(eas::DT_INT64);
        auto flat = outputs[i].flat<int64>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int64_val(flat(j));
        }
        break;
      }
      case DT_BOOL: {
        output.set_dtype(eas::DT_BOOL);
        auto flat = outputs[i].flat<bool>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_bool_val(flat(j));
        }
        break;
      }
      case DT_STRING: {
        output.set_dtype(eas::DT_STRING);
        auto flat = outputs[i].flat<std::string>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_string_val(flat(j));
        }
        break;
      }
      case DT_COMPLEX64: {
        output.set_dtype(eas::DT_COMPLEX64);
        auto flat = outputs[i].flat<complex64>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_float_val(flat(j).real());
          output.add_float_val(flat(j).imag());
        }
        break;
      }
      case DT_COMPLEX128: {
        output.set_dtype(eas::DT_COMPLEX128);
        auto flat = outputs[i].flat<complex128>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_double_val(flat(j).real());
          output.add_double_val(flat(j).imag());
        }
        break;
      }
      case DT_HALF: {
        output.set_dtype(eas::DT_HALF);
        auto flat = outputs[i].flat<Eigen::half>();
        for (int64 j = 0; j < total_dim_size; j++)
          output.add_float_val((float)flat(j));
        break;
      }
      case DT_BFLOAT16: {
        output.set_dtype(eas::DT_BFLOAT16);
        auto flat = outputs[i].flat<bfloat16>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++) {
          float value;
          BFloat16ToFloat(&flat(j), &value, 1);
          output.add_float_val(value);
        }
        break;
      }
      case tensorflow::eas::DT_RESOURCE: {
        LOG(ERROR) << "Output Tensor Not Support this DataType: DT_RESOURCE";
        break;
      }
      case tensorflow::eas::DT_VARIANT: {
        LOG(ERROR) << "Output Tensor Not Support this DataType: DT_VARIANT";
        break;
      }
      default:
        LOG(ERROR) << "Output Tensor Not Support this DataType";
        break;
    }
    (*response.mutable_outputs())[output_tensor_names[i]] = output;
  }
  return response;
}

}
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
    call.request.inputs.emplace_back(input.first, Proto2Tensor(input.second));
  }

  call.request.output_tensor_names =
      std::vector<std::string>(request.output_filter().begin(),
                               request.output_filter().end());

  return Status::OK();
}

Status ProtoBufParser::ParseResponseToBuf(const Call& call,
    void** output_data, int* output_size) {
  eas::PredictResponse response = Tensor2Response(call.request,
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
        call.request[i].inputs.emplace_back(input.first, Proto2Tensor(input.second));
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
      eas::PredictResponse response = Tensor2Response(call.request[i],
          call.response[i]);
      output_size[i] = response.ByteSize();
      output_data[i] = new char[*output_size];
      response.SerializeToArray(output_data[i], output_size[i]); 
    }
  };
  thread_pool_->ParallelFor(call.call_num, 10000, do_work);
  return Status::OK();
}

FlatBufferParser::FlatBufferParser(int thread_num) {
  thread_pool_.reset(new thread::ThreadPool(Env::Default(), "",
      thread_num));
}

} // processor
} // tensorflow

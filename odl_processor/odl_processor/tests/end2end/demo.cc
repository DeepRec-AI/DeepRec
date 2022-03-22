#include <iostream>
#include "odl_processor/serving/tf_processor.h"
#include "odl_processor/serving/tf_predict.pb.h"

static const char* model_config = "{ \
    \"omp_num_threads\": 4, \
    \"kmp_blocktime\": 0, \
    \"feature_store_type\": \"memory\", \
    \"serialize_protocol\": \"protobuf\", \
    \"inter_op_parallelism_threads\": 10, \
    \"intra_op_parallelism_threads\": 10, \
    \"init_timeout_minutes\": 1, \
    \"signature_name\": \"serving_default\", \
    \"read_thread_num\": 3, \
    \"update_thread_num\": 2, \
    \"model_store_type\": \"local\", \
    \"checkpoint_dir\": \"/tmp/checkpoint/\", \
    \"savedmodel_dir\": \"/tmp/saved_model/\" \
  } ";

int main(int argc, char** argv) {
  int state;
  void* model = initialize("", model_config, &state);
  if (state == -1) {
    std::cerr << "initialize error\n";
  }

  // input type: float
  ::tensorflow::eas::ArrayDataType dtype =
      ::tensorflow::eas::ArrayDataType::DT_FLOAT;
  // input shape: [1, 1]
  ::tensorflow::eas::ArrayShape array_shape;
  array_shape.add_dim(1);
  array_shape.add_dim(1);
  // input array
  ::tensorflow::eas::ArrayProto input;
  input.add_float_val(1.0);
  input.set_dtype(dtype);
  *(input.mutable_array_shape()) = array_shape;
  // PredictRequest
  ::tensorflow::eas::PredictRequest req;
  req.set_signature_name("serving_default");
  req.add_output_filter("y:0");
  (*req.mutable_inputs())["x:0"] = input;
  size_t size = req.ByteSizeLong(); 
  void *buffer = malloc(size);
  req.SerializeToArray(buffer, size);

  // do process
  void* output = nullptr;
  int output_size = 0;
  state = process(model, buffer, size, &output, &output_size);

  // parse response
  std::string output_string((char*)output, output_size);
  ::tensorflow::eas::PredictResponse resp;
  resp.ParseFromString(output_string);
  std::cout << "process returned state: " << state << ", response: " << resp.DebugString();

  return 0;
}


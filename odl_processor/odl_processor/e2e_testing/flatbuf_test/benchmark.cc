#include <random>
#include <chrono>   
#include <iostream>
#include <vector>
#include <string>
#include "odl_processor/e2e_testing/request_generated.h"
#include "odl_processor/serving/tf_predict.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"

using namespace std;
using namespace chrono;
using std::random_device;
using std::default_random_engine;

#define DEBUG 0
#define CONTENT_STRING_TYPE 0
#define FLAT_COPY_INPUT 1

static const int DIM_0 = 2;
static const int DIM_1 = 10000;

// for string type
static const int STR_LEN = 32; // 8, 16, 32

// Input tensor num
static const int COUNT = 10;

// testing count
static int TESTING_COUNT = 1000;

typedef int TFDataType;
typedef float TensorType; // using numeric type: float, int64_t
#define USE_STRING_TYPE 0   // using string tensor

namespace {

std::string StrRand(int length) {
  char tmp;
  string buffer;
  random_device rd;
  default_random_engine random(rd());

  for (int i = 0; i < length; i++) {
    tmp = random() % 36;
    if (tmp < 10) {
      tmp += '0';
    } else {
      tmp -= 10;
      tmp += 'A';
    }
    buffer += tmp;
  }

  return buffer;
}

}

template<typename VType>
void PrepareUserInputs(std::string& signature_name,
                       std::string& input_name,
                       TFDataType& data_type,
                       std::vector<long>& shape,
                       std::vector<VType>& content,
                       std::vector<std::string>& fetch_names) {
  signature_name = "default_serving";
  input_name = "input_from_feature_columns/fm_10169_embedding/const";
  data_type = 0; //TFDataType::FLOAT;
  // shape [2x10000]
  shape.push_back(DIM_0);
  shape.push_back(DIM_1);
  if (typeid(VType).name() == typeid(float).name()) {
    // DIM_0*DIM_1 floats
    const static float start_num = 1.22341;
    const static float incr_step = 0.67176;
    float x = start_num;
    content.reserve(DIM_0 * DIM_1);
    for (int i = 0; i < DIM_0 * DIM_1; ++i) {
      content.push_back(x);
      x += incr_step;
    }
  } else if (typeid(VType).name() == typeid(int64_t).name()) {
    // DIM_0*DIM_1 int64_t
    const static int64_t start_num = 18947;
    const static int64_t incr_step = 7659;
    int64_t x = start_num;
    content.reserve(DIM_0 * DIM_1);
    for (int i = 0; i < DIM_0 * DIM_1; ++i) {
      content.push_back(x);
      x += incr_step;
    }
  }

  fetch_names.push_back("fetch_0");
  fetch_names.push_back("fetch_1");
  fetch_names.push_back("fetch_2");
}

void PrepareUserStringInputs(std::string& signature_name,
                             std::string& input_name,
                             TFDataType& data_type,
                             std::vector<long>& shape,
                             std::vector<std::string>& content,
                             std::vector<std::string>& fetch_names) {
  signature_name = "default_serving";
  input_name = "input_from_feature_columns/fm_10169_embedding/const";
  data_type = 0; //TFDataType::FLOAT;
  shape.push_back(DIM_0);
  shape.push_back(DIM_1);
  // string type
  // DIM_0*DIM_1 strings
  // A string len = STR_LEN
  content.reserve(DIM_0 * DIM_1);
  for (int i = 0; i < DIM_0 * DIM_1; ++i) {
    content.push_back(StrRand(STR_LEN));
  }

  fetch_names.push_back("fetch_0");
  fetch_names.push_back("fetch_1");
  fetch_names.push_back("fetch_2");
}

// New User API
template<typename VType>
void EncodeByFlatBuffer(flatbuffers::FlatBufferBuilder& fbb,
                        const std::string& signature_name,
                        std::vector<std::string>& input_names,
                        std::vector<TFDataType>& data_types,
                        std::vector<std::vector<long>>& shapes,
                        std::vector<char*>& contents,
                        std::vector<char*>& aggregate_bufs,
                        std::vector<int>& string_tensor_lens,
                        std::vector<int>& each_tensor_total_len,
                        std::vector<std::string>& fetch_names) {
  // ---
  auto fsig_name = fbb.CreateString(signature_name);

  // ---
  std::vector<flatbuffers::Offset<flatbuffers::String>> tmp_input_names;
  for (auto& name : input_names) {
    tmp_input_names.push_back(fbb.CreateString(name));
  }
  auto finput_names = fbb.CreateVector(tmp_input_names);

  // ---
  auto fdata_types = fbb.CreateVector(data_types);

  // ---
  std::vector<flatbuffers::Offset<flatbuffers::Vector<int64_t>>> tmp_shape_vecs;
  for (size_t i = 0; i < shapes.size(); ++i) {
    auto vec_shape = fbb.CreateVector(shapes[i]);
    tmp_shape_vecs.push_back(vec_shape);
  }

  std::vector<flatbuffers::Offset<tensorflow::eas::test::ShapeType>> tmp_shapes;
  for (size_t i = 0; i < shapes.size(); ++i) {
    tensorflow::eas::test::ShapeTypeBuilder shape_builder(fbb);
    shape_builder.add_dim(tmp_shape_vecs[i]);
    tmp_shapes.push_back(shape_builder.Finish());
  }
  auto fshapes = fbb.CreateVector(tmp_shapes);

#if USE_STRING_TYPE
  // input0 -> buf0, input1 -> buf1 ...
  // count size0, size1 ... size_count; count size0, size1 ... size_count; ...
  auto ftensor_lens = fbb.CreateVector(string_tensor_lens); 
 
  std::vector<flatbuffers::Offset<flatbuffers::String>> string_aggregate_bufs;
  string_aggregate_bufs.reserve(aggregate_bufs.size());
  for (size_t i = 0; i < aggregate_bufs.size(); ++i) {
    string_aggregate_bufs.emplace_back(fbb.CreateString(aggregate_bufs[i],
                                                        each_tensor_total_len[i]));
  }
  auto fstr_content = fbb.CreateVector(string_aggregate_bufs);


#else
  // ---
  // float and int64_t type tensor
  #if CONTENT_STRING_TYPE
  // content: String type
  std::vector<flatbuffers::Offset<flatbuffers::String>> tmp_content;
  for (size_t i = 0; i < contents.size(); ++i) {
    size_t len = 1;
    for (size_t j = 0; j < shapes[i].size(); ++j) len *= shapes[i][j];
    tmp_content.push_back(fbb.CreateString(
        std::string(contents[i], len * sizeof(VType))));
  }
  auto fcontent = fbb.CreateVector(tmp_content);
  #else
  // content: [byte] type
  std::vector<flatbuffers::Offset<flatbuffers::Vector<int8_t>>> tmp_content_vecs;
  for (size_t i = 0; i < contents.size(); ++i) {
    size_t len = 1;
    for (size_t j = 0; j < shapes[i].size(); ++j) len *= shapes[i][j];
    len *= sizeof(VType);
    auto vec_content = fbb.CreateVector(
        reinterpret_cast<const int8_t*>(contents[i]), len);
    tmp_content_vecs.push_back(vec_content);
  }

  std::vector<flatbuffers::Offset<tensorflow::eas::test::ContentType>> tmp_contents;
  for (size_t i = 0; i < tmp_content_vecs.size(); ++i) {
    tensorflow::eas::test::ContentTypeBuilder content_builder(fbb);
    content_builder.add_content(tmp_content_vecs[i]);
    tmp_contents.push_back(content_builder.Finish());
  }

  auto fcontent = fbb.CreateVector(tmp_contents);

  #endif

#endif

  // ---
  std::vector<flatbuffers::Offset<flatbuffers::String>> tmp_fetch_names;
  for (auto& name : fetch_names) {
    tmp_fetch_names.push_back(fbb.CreateString(name));
  }
  auto ffetch_names = fbb.CreateVector(tmp_fetch_names);

  // -----------------

  tensorflow::eas::test::PredictRequestBuilder builder(fbb);
  builder.add_signature_name(fsig_name);
  builder.add_feed_names(finput_names);
  builder.add_types(fdata_types);
  builder.add_shapes(fshapes);
#if USE_STRING_TYPE
  builder.add_string_content_len(ftensor_lens);
  builder.add_string_content(fstr_content);
#else
  builder.add_content(fcontent);
#endif
  builder.add_fetch_names(ffetch_names);
  fbb.Finish(builder.Finish());
}

void EncodeStringTensorByProtoBuffer(tensorflow::eas::PredictRequest& req,
                                     const std::string& signature_name,
                                     std::string& input_name,
                                     TFDataType& data_type,
                                     std::vector<long>& shape,
                                     std::vector<std::string>& content,
                                     std::vector<std::string>& fetch_names) {
  req.set_signature_name(signature_name);

  for (int i = 0; i < COUNT; ++i) {
    tensorflow::eas::ArrayProto array_proto;
    array_proto.set_dtype(tensorflow::eas::ArrayDataType::DT_STRING);
    tensorflow::eas::ArrayShape *array_shape = array_proto.mutable_array_shape();
    for (std::vector<long>::const_iterator it = shape.begin(); it != shape.end();
         ++it) {
      array_shape->add_dim(*it);
    }
    *array_proto.mutable_string_val() = {content.begin(), content.end()};
    (*(req.mutable_inputs()))[input_name+"_"+std::to_string(i)] = array_proto;
  }

  for (auto name : fetch_names) {
    req.add_output_filter(name);
  }
}

template<typename VType>
void EncodeByProtoBuffer(tensorflow::eas::PredictRequest& req,
                         const std::string& signature_name,
                         std::string& input_name,
                         TFDataType& data_type,
                         std::vector<long>& shape,
                         std::vector<VType>& content,
                         std::vector<std::string>& fetch_names) {
  req.set_signature_name(signature_name);

  for (int i = 0; i < COUNT; ++i) {
    tensorflow::eas::ArrayProto array_proto;
    if (typeid(VType).name() == typeid(float).name()) {
      array_proto.set_dtype(tensorflow::eas::ArrayDataType::DT_FLOAT);
    } else if (typeid(VType).name() == typeid(int64_t).name()) {
      array_proto.set_dtype(tensorflow::eas::ArrayDataType::DT_INT64);
    }
    tensorflow::eas::ArrayShape *array_shape = array_proto.mutable_array_shape();
    for (std::vector<long>::const_iterator it = shape.begin(); it != shape.end();
         ++it) {
      array_shape->add_dim(*it);
    }
    if (typeid(VType).name() == typeid(float).name()) {
      *array_proto.mutable_float_val() = {content.begin(), content.end()};
    } else if (typeid(VType).name() == typeid(int64_t).name()) {
      *array_proto.mutable_int64_val() = {content.begin(), content.end()};
    }
    (*(req.mutable_inputs()))[input_name+"_"+std::to_string(i)] = array_proto;
  }

  for (auto name : fetch_names) {
    req.add_output_filter(name);
  }
}

tensorflow::Tensor Proto2Tensor(const tensorflow::eas::ArrayProto& input) {
  tensorflow::TensorShape tensor_shape;
  tensorflow::int64 total_size = 1;
  for (int i = 0; i < input.array_shape().dim_size(); ++i) {
    tensor_shape.AddDim(input.array_shape().dim(i));
    total_size *= input.array_shape().dim(i);
  }

  switch (input.dtype()) {
    case tensorflow::eas::DT_FLOAT: {
      if (total_size != input.float_val_size()) {
        LOG(FATAL) << "Invalid input.";
      }   
      tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensor_shape);
      auto flat = tensor.flat<float>();
      memcpy(flat.data(), input.float_val().data(),
          input.float_val_size() * sizeof(float));

      return tensor;
    }
    case tensorflow::eas::DT_INT64: {
      if (total_size != input.int64_val_size()) {
        LOG(FATAL) << "Invalid input.";
      }
      tensorflow::Tensor tensor(tensorflow::DT_INT64, tensor_shape);
      auto flat = tensor.flat<tensorflow::int64>();
      memcpy(flat.data(), input.int64_val().data(),
          input.int64_val_size() * sizeof(int64_t));

      return tensor;
    }
    case tensorflow::eas::DT_STRING: {
      if (total_size != input.string_val_size()) {
        LOG(FATAL) << "Invalid input.";
      }   
      tensorflow::Tensor tensor(tensorflow::DT_STRING, tensor_shape);
      auto flat = tensor.flat<std::string>();
      for (int i = 0; i < input.string_val_size(); i++) {
        flat(i) = input.string_val(i);
      }
      return tensor;
    }
    default: {
      LOG(FATAL) << "Input Tensor Not Support this DataType";
      break;
    }
  }

  return tensorflow::Tensor();
}

 
int main() {

  std::cout << "DIM_0: " << DIM_0
            << ", DIM_1: " << DIM_1
            << ", COUNT: " << COUNT << "\n";

#if DEBUG
  // if print debug log, only tesing once.
  TESTING_COUNT = 1;
#endif

  // user input
  std::string signature_name("default_serving");
  std::string input_name;
  TFDataType data_type;
  std::vector<long> shape;
  std::vector<TensorType> content;
  std::vector<std::string> string_content;
  std::vector<std::string> fetch_names;

  std::vector<std::string> input_names;
  std::vector<TFDataType> data_types;
  std::vector<std::vector<long>> shapes;
  std::vector<char*> contents;
  std::vector<std::vector<std::string>> string_contents;

#if USE_STRING_TYPE
  {
    PrepareUserStringInputs(signature_name, input_name, data_type,
                            shape, string_content, fetch_names);
  }
#else
  {
    // prepare data (will be used 10 times)
    PrepareUserInputs(signature_name, input_name, data_type,
                      shape, content, fetch_names);
  }
#endif

  for (int i = 0; i < COUNT; ++i) {
    input_names.push_back(input_name + "_" + std::to_string(i));
    data_types.push_back(data_type);
    shapes.push_back(shape);
#if USE_STRING_TYPE
    {
      string_contents.push_back(string_content);
    }
#else
    {
      contents.push_back((char*)content.data());
    }
#endif
  }

  // ------------------------ FLATBUFFER -----------------------------

  auto flat_start_timer = system_clock::now();

  // flatbuffer: testing TESTING_COUNT times
  //
  for (int x = 0; x < TESTING_COUNT; ++x) {
    // 1) encode - flatbuffer
    flatbuffers::FlatBufferBuilder fbb(25000);

    std::vector<char*> aggregate_bufs;
    std::vector<int> string_tensor_lens;
    std::vector<int> each_tensor_total_len;
    each_tensor_total_len.reserve(string_content.size());
    string_tensor_lens.reserve(string_content.size());

    #if FLAT_COPY_INPUT
      {
        #if USE_STRING_TYPE
        size_t total_len = 0;
        for (auto c : string_content) {
          total_len += c.length();
        }
        #endif
        // Use EAS client API
        std::vector<std::string> ninput_names;
        std::vector<std::vector<long>> nshapes;
        std::vector<TFDataType> ndata_types;
        std::vector<std::vector<TensorType>> ncontents;
        //std::vector<std::vector<std::string>> nstr_contents;
        ninput_names.reserve(COUNT);
        nshapes.reserve(COUNT);
        ndata_types.reserve(COUNT);
        ncontents.reserve(COUNT);
        for (int i = 0; i < COUNT; ++i) {
          ninput_names.push_back(input_name + "_" + std::to_string(i));
          nshapes.push_back(shape);
          ndata_types.push_back(data_type);
          #if not USE_STRING_TYPE
          ncontents.push_back(content);
          #else
          // 1)
          //fbb.StartFastCreateString(total_len);

          // push tensor count
          string_tensor_lens.push_back(string_content.size());
          char* buf = new char[total_len];
          size_t offset = 0;
          for (auto c : string_content) {
            // 2)
            //fbb.FastCreateString(c.c_str(), c.length());
            memcpy(buf + offset, c.c_str(), c.length());
            offset += c.length();
            // push each tensor length
            string_tensor_lens.push_back(c.length());
          }
          // 3)
          //string_aggregate_buf_offsets.push_back(fbb.EndFastCreateString(total_len));

          each_tensor_total_len.push_back(offset);
          aggregate_bufs.push_back(buf);
          #endif
        }
      }
      EncodeByFlatBuffer<TensorType>(
          fbb, signature_name, input_names, data_types,
          shapes, contents, aggregate_bufs, string_tensor_lens,
          each_tensor_total_len, fetch_names);
    #else
      // Should create a new EAS client API
      EncodeByFlatBuffer<TensorType>(
          fbb, signature_name, input_names, data_types,
          shapes, contents, aggregate_bufs, string_tensor_lens,
          each_tensor_total_len, fetch_names);
    #endif
    std::string wrapper_data((char*)fbb.GetBufferPointer(), fbb.GetSize());

    // 2) decode - flatbuffer
    const tensorflow::eas::test::PredictRequest* flat_recv_req =
        flatbuffers::GetRoot<tensorflow::eas::test::PredictRequest>((void*)(wrapper_data.data()));

    #if USE_STRING_TYPE
    auto& string_content_len_arr = (*(flat_recv_req->string_content_len()));
    size_t index = 0;
    #endif

    int count = flat_recv_req->shapes()->size();
    for (int i = 0; i < count; ++i) {
      // shape
      tensorflow::TensorShape tensor_shape;
      int total_dim_count = 1;
      for (auto d : *((*(flat_recv_req->shapes()))[i]->dim())) {
        tensor_shape.AddDim(d);
        total_dim_count *= d;
      }
 
#if USE_STRING_TYPE
      // String Tensor
      tensorflow::Tensor t(tensorflow::DT_STRING, tensor_shape);
      auto flat = t.flat<std::string>();
      int tensor_count = string_content_len_arr[index++];
      if (tensor_count != total_dim_count) {
        assert(false && "tensor_count != total_dim_count");
      }

      size_t offset = 0;
      const char* curr_input_tensor_buf = (*(flat_recv_req->string_content()))[i]->c_str();
      for (int x = 0 ; x < tensor_count; ++x) {
        flat(x) = std::move(std::string(curr_input_tensor_buf + offset, string_content_len_arr[index]));
        offset += string_content_len_arr[index++];
      }

      #if DEBUG
      std::cout << t.DebugString() << "\n";
      #endif
#else
      // type: DT_FLOAT
      // tensor
      /*
      // TODO OPTIMIZATION: tf1.15 can create tensor from a buffer.
      //
      tensorflow::TensorBuffer tbuffer((void*)((*(flat_recv_req->content()))[i]->c_str()));
      tensorflow::Tensor t(tensorflow::DT_FLOAT, tensor_shape, &tbuffer);
      */
      if (typeid(TensorType).name() == typeid(float).name()) {
        tensorflow::Tensor t(tensorflow::DT_FLOAT, tensor_shape);
        auto flat = t.flat<float>();
        #if CONTENT_STRING_TYPE
          memcpy(flat.data(), (*(flat_recv_req->content()))[i]->c_str(),
                 (*(flat_recv_req->content()))[i]->size());
        #else
          memcpy(flat.data(), (*(flat_recv_req->content()))[i]->content()->Data(),
                 (*(flat_recv_req->content()))[i]->content()->size());
        #endif

        //std::cout << t.DebugString() << "\n";
      } else if (typeid(TensorType).name() == typeid(int64_t).name()) {
        tensorflow::Tensor t(tensorflow::DT_INT64, tensor_shape);
        auto flat = t.flat<tensorflow::int64>();
        #if CONTENT_STRING_TYPE
          memcpy(flat.data(), (*(flat_recv_req->content()))[i]->c_str(),
                 (*(flat_recv_req->content()))[i]->size());
        #else
          memcpy(flat.data(), (*(flat_recv_req->content()))[i]->content()->Data(),
                 (*(flat_recv_req->content()))[i]->content()->size());
        #endif

        //std::cout << t.DebugString() << "\n";
      }
      
#endif
    }

  #if DEBUG
    std::cout << "signature_name = " << flat_recv_req->signature_name()->str() << "\n";
    int idx = 0;
    for (auto fname : *(flat_recv_req->feed_names())) {
      std::cout << "#" << idx++ << "_feed_name = " << fname->str() << "\n";
    }
    idx = 0;
    for (auto t : *(flat_recv_req->types())) {
      std::cout << "#" << idx++ << "_type = " << t << "\n";
    }
    idx = 0;
    for (auto s : *(flat_recv_req->shapes())) {
      std::cout << "#" << idx++ << "_dim = ";
      for (auto d : *(s->dim())) {
        std::cout << d << " ";
      }
      std::cout << "\n";
    }
    idx = 0;
    for (auto fname : *(flat_recv_req->fetch_names())) {
      std::cout << "#" << idx++ << "_fetch_name = " << fname->str() << "\n";
    }
    idx = 0;
    #if not USE_STRING_TYPE
    #if CONTENT_STRING_TYPE
      for (auto c : *(flat_recv_req->content())) {
        TensorType* buf = (TensorType*)c->c_str();
        size_t total_len = c->size();
        size_t count = total_len / sizeof(TensorType);
        std::cout << "#" << idx++ << "_num count = " << count << "\n";
        for (size_t i = 0; i < count; ++i) {
          std::cout << (TensorType)*(buf + i) << " ";
        }
        std::cout << "\n";
      }
    #else
      for (auto c : *(flat_recv_req->content())) {
        TensorType* buf = (TensorType*)(c->content()->Data());
        size_t total_len = c->content()->size();
        size_t count = total_len / sizeof(TensorType);
        std::cout << "#" << idx++ << "_num count = " << count << "\n";
        for (size_t i = 0; i < count; ++i) {
          std::cout << (TensorType)*(buf + i) << " ";
        }
        std::cout << "\n";
      }
    #endif
    #endif

  #endif 
  }

  auto flat_end_timer = system_clock::now();

  // ------------------------- PROTOBUF ----------------------------

  auto pb_start_timer = system_clock::now();
  
  // protobuf: testing TESTING_COUNT times
  //
  for (int x = 0; x < TESTING_COUNT; ++x) {
    // 1) encode - protobuf
    tensorflow::eas::PredictRequest pb_req;
#if USE_STRING_TYPE
    EncodeStringTensorByProtoBuffer(pb_req, signature_name, input_name, data_type,
                                    shape, string_content, fetch_names);
#else
    EncodeByProtoBuffer<TensorType>(pb_req, signature_name, input_name, data_type,
                                    shape, content, fetch_names);
#endif

    std::string pb_req_str;
    pb_req.SerializeToString(&pb_req_str);

    // 2) decode - protobufbuffer
    tensorflow::eas::PredictRequest pb_recv_req;
    pb_recv_req.ParseFromArray(pb_req_str.c_str(), pb_req_str.size());
    for (auto& input : pb_recv_req.inputs()) {
      tensorflow::Tensor t = Proto2Tensor(input.second);
    }

  #if DEBUG
    std::cout << "sig_name: " << pb_recv_req.signature_name() << "\n";
    for (auto& input : pb_recv_req.inputs()) {
      std::cout << "name: " << input.first
                << ", tensor: " << Proto2Tensor(input.second).DebugString() << "\n";
    }
    for (auto name : pb_recv_req.output_filter()) {
      std::cout << "fetch_name: " << name << "\n";
    }
  #endif
  }

  auto pb_end_timer = system_clock::now();

  // -----------------------------------------------------

  // encode - raw
  //EncodeByRaw(input_name, data_type, shape, content);

  // decode - raw

  // --------------------- PRINT -------------------------

  auto flat_duration = duration_cast<microseconds>(flat_end_timer - flat_start_timer);
  auto pb_duration = duration_cast<microseconds>(pb_end_timer - pb_start_timer);
  std::cout << "flat: "
            << double(flat_duration.count()) * microseconds::period::num / microseconds::period::den << "\n";
  std::cout << "pb: "
            << double(pb_duration.count()) * microseconds::period::num / microseconds::period::den << "\n";

  return 0;
}


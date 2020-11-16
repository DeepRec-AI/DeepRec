#include "run_predict.h"

#include "tensorflow/core/framework/tensor.h"

using tensorflow::ERROR;
using tensorflow::TensorShape;

RunRequest::~RunRequest() {
  auto iter = inputs.begin();
  for (; iter != inputs.end(); iter++) {
    delete iter->second;
  }
  inputs.clear();
  outputAliasNames.clear();
}

RunRequest::RunRequest() {
  inputs.clear();
  outputAliasNames.clear();
}

void RunRequest::SetSignatureName(const std::string& value) {
  signatureName = value;
}

// DT_FLOAT
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<float>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* floatTensor = new Tensor(tensorflow::DT_FLOAT, tensorShape);
  auto floatTensorFat = floatTensor->flat<float>();
  memcpy(floatTensorFat.data(), &content[0], content.size() * sizeof(float));
  inputs.emplace(inputAliasName, floatTensor);
}

// DT_DOUBLE
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<double>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* doubleTensor = new Tensor(tensorflow::DT_DOUBLE, tensorShape);
  auto doubleTensorFat = doubleTensor->flat<double>();
  memcpy(doubleTensorFat.data(), &content[0], content.size() * sizeof(double));
  inputs.insert(std::pair<std::string, Tensor*>(inputAliasName, doubleTensor));
}

// DT_INT32
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<int>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* intTensor = new Tensor(tensorflow::DT_INT32, tensorShape);
  auto intTensorFat = intTensor->flat<int>();
  memcpy(intTensorFat.data(), &content[0], content.size() * sizeof(int));
  inputs.insert(std::pair<std::string, Tensor*>(inputAliasName, intTensor));
}

// DT_UINT8
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<unsigned char>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* uint8Tensor = new Tensor(tensorflow::DT_UINT8, tensorShape);
  auto uint8TensorFat = uint8Tensor->flat<unsigned char>();
  memcpy(uint8TensorFat.data(), &content[0],
         content.size() * sizeof(unsigned char));
  inputs.insert(std::pair<std::string, Tensor*>(inputAliasName, uint8Tensor));
}

// DT_INT16
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<short>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* int16Tensor = new Tensor(tensorflow::DT_INT16, tensorShape);
  auto int16TensorFat = int16Tensor->flat<short>();
  memcpy(int16TensorFat.data(), &content[0], content.size() * sizeof(short));
  inputs.insert(std::pair<std::string, Tensor*>(inputAliasName, int16Tensor));
}

// DT_INT8
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<signed char>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* int8Tensor = new Tensor(tensorflow::DT_INT8, tensorShape);
  auto int8TensorFat = int8Tensor->flat<signed char>();
  memcpy(int8TensorFat.data(), &content[0],
         content.size() * sizeof(signed char));
  inputs.insert(std::pair<std::string, Tensor*>(inputAliasName, int8Tensor));
}

// DT_STRING
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<std::string>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* stringTensor = new Tensor(tensorflow::DT_STRING, tensorShape);
  auto stringTensorFat = stringTensor->flat<std::string>();
  for (int i = 0; i < content.size(); i++) {
    stringTensorFat(i) = content[i];
  }
  inputs.insert(std::pair<std::string, Tensor*>(inputAliasName, stringTensor));
}

// DT_COMPLEX64
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<std::complex<float> >& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* complex64Tensor = new Tensor(tensorflow::DT_COMPLEX64, tensorShape);
  auto complex64TensorFat = complex64Tensor->flat<std::complex<float> >();
  for (int i = 0; i < content.size(); i++) {
    complex64TensorFat(i) = content[i];
  }
  inputs.insert(
      std::pair<std::string, Tensor*>(inputAliasName, complex64Tensor));
}

// DT_INT64
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<long long>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* longTensor = new Tensor(tensorflow::DT_INT64, tensorShape);
  auto longTensorFat = longTensor->flat<long long>();
  memcpy(longTensorFat.data(), &content[0], content.size() * sizeof(long long));
  inputs.insert(std::pair<std::string, Tensor*>(inputAliasName, longTensor));
}

// DT_BOOL
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<bool>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* boolTensor = new Tensor(tensorflow::DT_BOOL, tensorShape);
  auto boolTensorFat = boolTensor->flat<bool>();
  for (int i = 0; i < content.size(); i++) boolTensorFat(i) = content[i];
  inputs.insert(std::pair<std::string, Tensor*>(inputAliasName, boolTensor));
}

// DT_UINT16
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<unsigned short>& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* uint16Tensor = new Tensor(tensorflow::DT_UINT16, tensorShape);
  auto uint16TensorFat = uint16Tensor->flat<unsigned short>();
  memcpy(uint16TensorFat.data(), &content[0],
         content.size() * sizeof(unsigned short));
  inputs.insert(std::pair<std::string, Tensor*>(inputAliasName, uint16Tensor));
}

// DT_COMPLEX128
void RunRequest::AddFeed(const std::string& inputAliasName,
                         std::vector<long long>& shape,
                         std::vector<std::complex<double> >& content) {
  TensorShape tensorShape;
  long long total_dim_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    tensorShape.AddDim(shape[i]);
    total_dim_size *= shape[i];
  }
  if (total_dim_size != content.size()) {
    LOG(ERROR) << "The shape and content of the input tensor " << inputAliasName
               << " are not the same size";
    exit(-1);
  }
  Tensor* complex128Tensor = new Tensor(tensorflow::DT_COMPLEX128, tensorShape);
  auto complex128TensorFat = complex128Tensor->flat<std::complex<double> >();
  for (int i = 0; i < content.size(); i++) {
    complex128TensorFat(i) = content[i];
  }
  inputs.insert(
      std::pair<std::string, Tensor*>(inputAliasName, complex128Tensor));
}

void RunRequest::AddFetch(const std::string& outputAliasName) {
  outputAliasNames.emplace_back(outputAliasName);
}

const std::string& RunRequest::GetSignatureName() const {
  return signatureName;
}

const std::vector<std::string>& RunRequest::GetOutputAliasNames() const {
  return outputAliasNames;
}

const std::map<std::string, Tensor*>& RunRequest::GetInputs() const {
  return inputs;
}

RunResponse::~RunResponse() {
  auto iter = outputs.begin();
  for (; iter != outputs.end(); iter++) {
    delete iter->second;
  }
  outputs.clear();
}

RunResponse::RunResponse() { outputs.clear(); }

void RunResponse::GetTensorShape(const std::string& outputAliasName,
                                 std::vector<long long>* shape) {
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  int dimSize = iter->second->dims();
  shape->reserve(dimSize);
  for (int i = 0; i < dimSize; i++) {
    shape->push_back(iter->second->dim_size(i));
  }
}

void RunResponse::SetOutputTensor(const std::string& key, const Tensor& value) {
  Tensor* outputTensor = new Tensor(value);
  outputs.insert(std::pair<std::string, Tensor*>(key, outputTensor));
}

// DT_FLOAT
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<float>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto floatTensorFat = iter->second->flat<float>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(floatTensorFat(i));
}
// DT_DOUBLE
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<double>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto doubleTensorFat = iter->second->flat<double>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(doubleTensorFat(i));
}
// DT_INT32
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<int>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto intTensorFat = iter->second->flat<int>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(intTensorFat(i));
}
// DT_UINT8
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<unsigned char>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto uint8TensorFat = iter->second->flat<unsigned char>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(uint8TensorFat(i));
}
// DT_INT16
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<short>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto int16TensorFat = iter->second->flat<short>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(int16TensorFat(i));
}
// DT_INT8
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<signed char>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto int8TensorFat = iter->second->flat<signed char>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(int8TensorFat(i));
}
// DT_STRING
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<std::string>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto stringTensorFat = iter->second->flat<std::string>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(stringTensorFat(i));
}
// DT_COMPLEX64
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<std::complex<float> >* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto complex64TensorFat = iter->second->flat<std::complex<float> >();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++)
    result->push_back(complex64TensorFat(i));
}
// DT_INT64
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<long long>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto longTensorFat = iter->second->flat<long long>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(longTensorFat(i));
}
// DT_BOOL
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<bool>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto boolTensorFat = iter->second->flat<bool>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(boolTensorFat(i));
}
// DT_UINT16
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<unsigned short>* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto uint16TensorFat = iter->second->flat<unsigned short>();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++) result->push_back(uint16TensorFat(i));
}
// DT_COMPLEX128
void RunResponse::GetData(const std::string& outputAliasName,
                          std::vector<std::complex<double> >* result) {
  int totalDimSize = 1;
  auto iter = outputs.find(outputAliasName);
  if (iter == outputs.end()) {
    LOG(ERROR) << "Not Found Tensor Alias Name: " << outputAliasName;
    exit(-1);
  }
  for (int i = 0; i < iter->second->dims(); i++)
    totalDimSize *= iter->second->dim_size(i);
  auto complex128TensorFat = iter->second->flat<std::complex<double> >();
  result->reserve(totalDimSize);
  for (int i = 0; i < totalDimSize; i++)
    result->push_back(complex128TensorFat(i));
}

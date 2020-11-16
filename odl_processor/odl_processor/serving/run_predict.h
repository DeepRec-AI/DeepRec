#ifndef TENSORFLOW_RUN_PREDICT_H
#define TENSORFLOW_RUN_PREDICT_H
#include <complex>
#include <map>
#include <string>
#include <vector>
namespace tensorflow {
class Tensor;
}  // namespace tensorflow

using tensorflow::Tensor;
class RunRequest {
 public:
  ~RunRequest();
  RunRequest();
  void SetSignatureName(const std::string& value);
  // DT_FLOAT
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<float>& content);
  // DT_DOUBLE
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<double>& content);
  // DT_INT32
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<int>& content);
  // DT_UINT8
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<unsigned char>& content);
  // DT_INT16
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<short>& content);
  // DT_INT8
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<signed char>& content);
  // DT_STRING
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<std::string>& content);
  // DT_COMPLEX64
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<std::complex<float> >& content);
  // DT_INT64
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<long long>& content);
  // DT_BOOL
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<bool>& content);
  // DT_UINT16
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<unsigned short>& content);
  // DT_COMPLEX128
  void AddFeed(const std::string& inputAliasName, std::vector<long long>& shape,
               std::vector<std::complex<double> >& content);
  void AddFetch(const std::string& outputAliasName);

  const std::string& GetSignatureName() const;
  const std::vector<std::string>& GetOutputAliasNames() const;
  const std::map<std::string, Tensor*>& GetInputs() const;

 private:
  std::map<std::string, Tensor*> inputs;
  std::vector<std::string> outputAliasNames;
  std::string signatureName;
};

class RunResponse {
 public:
  ~RunResponse();
  RunResponse();
  void GetTensorShape(const std::string& outputAliasName,
                      std::vector<long long>*);
  void SetOutputTensor(const std::string&, const Tensor&);
  // DT_FLOAT
  void GetData(const std::string& outputAliasName, std::vector<float>*);
  // DT_DOUBLE
  void GetData(const std::string& outputAliasName, std::vector<double>*);
  // DT_INT32
  void GetData(const std::string& outputAliasName, std::vector<int>*);
  // DT_UINT8
  void GetData(const std::string& outputAliasName, std::vector<unsigned char>*);
  // DT_INT16
  void GetData(const std::string& outputAliasName, std::vector<short>*);
  // DT_INT8
  void GetData(const std::string& outputAliasName, std::vector<signed char>*);
  // DT_STRING
  void GetData(const std::string& outputAliasName, std::vector<std::string>*);
  // DT_COMPLEX64
  void GetData(const std::string& outputAliasName,
               std::vector<std::complex<float> >*);
  // DT_INT64
  void GetData(const std::string& outputAliasName, std::vector<long long>*);
  // DT_BOOL
  void GetData(const std::string& outputAliasName, std::vector<bool>*);
  // DT_UINT16
  void GetData(const std::string& outputAliasName,
               std::vector<unsigned short>*);
  // DT_COMPLEX128
  void GetData(const std::string& outputAliasName,
               std::vector<std::complex<double> >*);

 private:
  std::map<std::string, Tensor*> outputs;
};
#endif

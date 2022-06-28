#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/public/session.h"
namespace tensorflow {
namespace test_utils {
inline string datatypeToString(tensorflow::DataType dtype) {
    if (dtype == DataType::DT_FLOAT) return "FLOAT";
    if (dtype == DataType::DT_BFLOAT16) return "BFLOAT16";
    return "UNKNOWN";
}

template <typename T>
string vectorToString(std::vector<T> vec) {
    if (vec.size() == 0) return "INCORRECT";
    std::ostringstream result;
    for (int i=0; i < vec.size() - 1; i++) {
        result <<  vec[i] << "_";
    }
    result << vec[vec.size()-1];
    return result.str();
}

inline void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                        Tensor* output) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(
    tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph));
    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));
    *output = unfused_tensors[0];
}

template<typename T>
Tensor makeTensor(std::vector<long long int> tensorShape, DataType dataType, float min_val, float max_val) {
    auto outputTensor = Tensor(dataType, TensorShape(tensorflow::gtl::ArraySlice<long long int>(tensorShape.data(), tensorShape.size())));
    auto range = max_val - min_val;
    outputTensor.flat<T>() = outputTensor.flat<T>().template setRandom<Eigen::internal::UniformRandomGenerator<T>>(); // (0, 1)
    outputTensor.flat<T>() = outputTensor.flat<T>() * outputTensor.flat<T>().constant(static_cast<T>(range)); // (0, range)
    outputTensor.flat<T>() = outputTensor.flat<T>() + outputTensor.flat<T>().constant(static_cast<T>(min_val)); // (min_val, max_val)
    return outputTensor;
}

inline Tensor makeTensor(std::vector<long long int> tensorShape, DataType dataType, float min_val, float max_val) {
        switch(dataType) {
            case DT_FLOAT:
                return makeTensor<float>(tensorShape, dataType, min_val, max_val);
            case DT_BFLOAT16:
                return makeTensor<Eigen::bfloat16>(tensorShape, dataType, min_val, max_val);
            default:
                return Tensor();
        }
} 
} // namespace utils
} // namespace tensorflow
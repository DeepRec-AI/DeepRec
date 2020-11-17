#ifndef TENSORFLOW_SERVING_SAVED_MODEL_H
#define TENSORFLOW_SERVING_SAVED_MODEL_H
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tensorflow {
class SavedModelBundle;
class Tensor;
class SignatureDef;
namespace serving {
class SessionBundleConfig;
}  // namespace serving
namespace eas {
class PredictRequest;
class PredictResponse;
}  // namespace eas
}  // namespace tensorflow
namespace processor {
class FreezeBundle;
}  // namespace eas
class RunRequest;
class RunResponse;

using processor::FreezeBundle;
using tensorflow::Tensor;
using tensorflow::SignatureDef;
using tensorflow::SavedModelBundle;
using tensorflow::serving::SessionBundleConfig;
using tensorflow::eas::PredictRequest;
using tensorflow::eas::PredictResponse;

namespace processor {
struct Signature {
  std::shared_ptr<SignatureDef> signature_def;
  std::string signature_name;

  Signature(SignatureDef* sig_def, const std::string& name);
};
}

class SavedModelLoader {
 public:
  ~SavedModelLoader();
  SavedModelLoader(int inter_op_parallelism_threads = 0,
                   int intra_op_parallelism_threads = 0,
                   double per_process_gpu_memory_fraction = 1.0,
                   bool allow_growth = true, bool allow_soft_placement = true,
                   bool log_device_placement = false,
                   bool allow_bfc_allocator = true);
  int LoadModel(const std::string&);
  int Predict(const PredictRequest&, PredictResponse*);
  int Predict(const RunRequest&, RunResponse*);
  processor::Signature* GetModelSignatureInfo();

 private:
  int SavedModelPredict(const PredictRequest&, PredictResponse*);
  int FreezeModelPredict(const PredictRequest&, PredictResponse*);
  int SavedModelPredict(const RunRequest&, RunResponse*);
  int FreezeModelPredict(const RunRequest&, RunResponse*);
  int PreProcessPrediction(const SignatureDef&, const PredictRequest&,
                           std::vector<std::pair<std::string, Tensor> >*,
                           std::vector<std::string>*,
                           std::vector<std::string>*);
  int ConvertProtoToTensor(std::vector<std::pair<std::string, Tensor> >*,
                           const std::string&, const PredictRequest&,
                           const std::string&);
  int ConvertTensorToProto(std::vector<Tensor>&, std::vector<std::string>&,
                           PredictResponse*);

 private:
  SavedModelBundle* saved_model_bundle;
  FreezeBundle* freeze_bundle;
  SessionBundleConfig* config;
  bool is_saved_model;
  bool is_freeze_model;
  processor::Signature* model_signature_info;
};

#endif

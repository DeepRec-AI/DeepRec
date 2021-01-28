#include "gtest/gtest.h"
#include "odl_processor/storage/feature_store_mgr.h"
#include "odl_processor/serving/model_session.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/serving/model_message.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace processor {
namespace {
ModelConfig CreateValidModelConfig() {
  ModelConfig config;
  config.checkpoint_dir = "oss://test_checkpoint/";
  config.savedmodel_dir = "oss://test_savedmodel/";

  config.signature_name = "tensorflow_serving";
  config.init_timeout_minutes = 2;

  config.inter_threads = 4;
  config.intra_threads = 2;

  config.feature_store_type = "local_redis";
  config.read_thread_num = 2;
  config.update_thread_num = 2;

  config.model_store_type = "oss";
  config.oss_endpoint = "";
  config.oss_access_id = "";
  config.oss_access_key = "";
  return config;
}
}

class ModelSessionMgrTest : public ::testing::Test {
};

class FakeSession : public Session {
 public:
  Status Create(const GraphDef& graph) override {
    return Status::OK();
  }

  Status Extend(const GraphDef& graph) override {
    return Status::OK();
  }

  Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override {
    return Status::OK();
  }

  Status Create(const RunOptions& run_options,
      const GraphDef& graph) override {
    return Status::OK();
  }

  Status Extend(const RunOptions& run_options,
      const GraphDef& graph) override {
    return Status::OK();
  }

  Status Close(const RunOptions& run_options) override {
    return Status::OK();
  }

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override {
    return Status::OK();
  }

  Status PRunSetup(const std::vector<string>& input_names,
                   const std::vector<string>& output_names,
                   const std::vector<string>& target_nodes,
                   string* handle) override {
    return Status::OK();
  }

  Status ListDevices(std::vector<DeviceAttributes>* response) override {
    return Status::OK();
  }

  Status Close() override {
    return Status::OK();
  }

  Status LocalDeviceManager(const DeviceMgr** output) override {
    return Status::OK();
  }

  Status MakeCallable(const CallableOptions& callable_options,
                      CallableHandle* out_handle) override {
    return Status::OK();
  }

  Status RunCallable(CallableHandle handle,
                     const std::vector<Tensor>& feed_tensors,
                     std::vector<Tensor>* fetch_tensors,
                     RunMetadata* run_metadata) override {
    return Status::OK();
  }

  Status ReleaseCallable(CallableHandle handle) override {
    return Status::OK();
  }
};

class FakeFeatureStoreMgr : public IFeatureStoreMgr {
 public:
  FakeFeatureStoreMgr(ModelConfig* config) {
  }

  Status GetValues(uint64_t model_version,
                   uint64_t feature2id,
                   const char* const keys,
                   char* const values,
                   size_t bytes_per_key,
                   size_t bytes_per_values,
                   size_t N,
                   const char* default_value,
                   BatchGetCallback cb) override {
    return Status::OK();
  }

  Status SetValues(uint64_t model_version,
                   uint64_t feature2id,
                   const char* const keys,
                   const char* const values,
                   size_t bytes_per_key,
                   size_t bytes_per_values,
                   size_t N,
                   BatchSetCallback cb) override {
    return Status::OK();
  }

  Status Reset() override {
    return Status::OK();
  }

  Status GetStorageMeta(StorageMeta* meta) {
    return Status::OK();
  }

  void GetStorageOptions(StorageMeta& meta,
                         StorageOptions** cur_opt,
                         StorageOptions** bak_opt) {
  }

  Status SetStorageActiveStatus(bool active) {
    return Status::OK();
  }

  Status GetModelVersion(int64_t* full_version,
                         int64_t* latest_version) {
    return Status::OK();
  }

  Status SetModelVersion(int64_t full_version,
                         int64_t latest_version) {
    return Status::OK();
  }

  Status GetStorageLock(int value, int timeout,
                        bool* success) {
    return Status::OK();
  }

  Status ReleaseStorageLock(int value) {
    return Status::OK();
  }
};

class TestableModelSessionMgr : public ModelSessionMgr {
 public:
  TestableModelSessionMgr(const MetaGraphDef& meta_graph_def,
      SessionOptions* sess_options, RunOptions* run_options) :
    ModelSessionMgr(meta_graph_def, sess_options, run_options) {}

  Status CreateSession(Session** sess) override {
    *sess = new FakeSession();
    return Status::OK();
  }

  size_t GetModelSessionSize() {
    if (serving_session_ != nullptr) {
      return sessions_.size() + 1;
    } else {
      return sessions_.size();
    }
  }

  void AddModelSession(IFeatureStoreMgr* sparse_storage) {
    Session* sess = nullptr;
    CreateSession(&sess);
    sessions_.emplace_back(new ModelSession(sess, Version(), sparse_storage));
  }

  void* GetServingSession() {
    return serving_session_;
  }

  Status RunRestoreOps(
      const char* ckpt_name, int64 full_ckpt_version,
      const char* savedmodel_dir, Session* session,
      IFeatureStoreMgr* sparse_storage, bool is_incr_ckpt,
      bool update_sparse, int64_t latest_version) override {
    return Status::OK();
  }
};

TEST_F(ModelSessionMgrTest, CreateModelSessionReturnStatusOK) {
  MetaGraphDef test_graph_def;
  SessionOptions sess_options;
  RunOptions run_options;
  TestableModelSessionMgr mgr(test_graph_def, &sess_options, &run_options);

  Version version;
  ModelConfig config = CreateValidModelConfig();
  FakeFeatureStoreMgr feature_store(&config);
  EXPECT_TRUE(mgr.CreateModelSession(version, config.checkpoint_dir.c_str(),
        &feature_store, false, false, &config).ok());
}

TEST_F(ModelSessionMgrTest, CreateModelSessionWhenPrevServingDone) {
  MetaGraphDef test_graph_def;
  SessionOptions sess_options;
  RunOptions run_options;
  TestableModelSessionMgr mgr(test_graph_def, &sess_options, &run_options);

  ModelConfig config = CreateValidModelConfig();
  FakeFeatureStoreMgr feature_store(&config);
  
  Version version_0;
  EXPECT_TRUE(mgr.CreateModelSession(version_0, config.checkpoint_dir.c_str(),
        &feature_store, false, false, &config).ok());
  
  EXPECT_EQ(1, mgr.GetModelSessionSize());
  auto prev_serving_session = mgr.GetServingSession();

  Request req;
  Response resp;
  EXPECT_TRUE(mgr.Predict(req, resp).ok());

  EXPECT_TRUE(mgr.CreateModelSession(version_0, config.checkpoint_dir.c_str(),
        &feature_store, false, false, &config).ok());

  EXPECT_EQ(1, mgr.GetModelSessionSize());

  EXPECT_TRUE(prev_serving_session != mgr.GetServingSession());
}

TEST_F(ModelSessionMgrTest, CleanupModelSessionWhenNoRequest) {
  MetaGraphDef test_graph_def;
  SessionOptions sess_options;
  RunOptions run_options;
  TestableModelSessionMgr mgr(test_graph_def, &sess_options, &run_options);
  
  ModelConfig config = CreateValidModelConfig();
  FakeFeatureStoreMgr feature_store(&config);
  
  Version version_0;
  EXPECT_TRUE(mgr.CreateModelSession(version_0, config.checkpoint_dir.c_str(),
        &feature_store, false, false, &config).ok());
  
  EXPECT_EQ(1, mgr.GetModelSessionSize());

  mgr.AddModelSession(&feature_store);
  mgr.AddModelSession(&feature_store);

  EXPECT_EQ(3, mgr.GetModelSessionSize());

  mgr.CleanupModelSession();

  EXPECT_EQ(1, mgr.GetModelSessionSize());
}

} // processor
} // tensorflow

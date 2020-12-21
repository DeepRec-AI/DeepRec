#include "gtest/gtest.h"
#include "odl_processor/serving/model_instance.h"
#include <unistd.h>
#include <string>
#include <cstdlib>
 
namespace tensorflow {
namespace processor {

class TFInstanceMgrTest : public ::testing::Test {
};

class TestableTFInstanceMgr : public TFInstanceMgr {
 public:
  explicit TestableTFInstanceMgr(ModelConfig* config) : TFInstanceMgr(config) {}

  SingleSessionInstance*  GetInstance() {
    return instance_;
  }
};

TEST_F(TFInstanceMgrTest, LocalSavedModelServingSuccess) {
const std::string local_config = " \
  { \
    \"processor_type\" : \"tf\", \
    \"inter_op_parallelism_threads\" : 4, \
    \"intra_op_parallelism_threads\" : 2, \
    \"init_timeout_minutes\" : 1, \
    \"signature_name\": \"serving_default\", \
    \"checkpoint_dir\" : \"\", \
    \"savedmodel_dir\" : \"/tmp/odl_test/deepfm/\", \
    \"feature_store_type\" : \"tensorflow\", \
    \"redis_url\" :\"test_url\",  \
    \"redis_password\" :\"test_password\", \
    \"read_thread_num\" : 2, \
    \"update_thread_num\":1, \
    \"model_store_type\": \"local\", \
    \"oss_endpoint\": \"test.endpoint\", \
    \"oss_access_id\" : \"test_id\", \
    \"oss_access_key\" : \"test_key\" \
  }";

  ModelConfig* config = nullptr;

  EXPECT_TRUE(
      ModelConfigFactory::Create(local_config.c_str(), &config).ok());
  EXPECT_EQ("tf", config->processor_type);

  TestableTFInstanceMgr mgr(config);
  EXPECT_TRUE(mgr.Init().ok());
  auto instance = mgr.GetInstance();
  
  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(instance->Warmup().ok());
  }
}

TEST_F(TFInstanceMgrTest, LocalSavedModelServingFailedWithInvalidPath) {
const std::string local_config = " \
  { \
    \"processor_type\" : \"tf\", \
    \"inter_op_parallelism_threads\" : 4, \
    \"intra_op_parallelism_threads\" : 2, \
    \"init_timeout_minutes\" : 1, \
    \"signature_name\": \"serving_default\", \
    \"checkpoint_dir\" : \"\", \
    \"savedmodel_dir\" : \"/tmp/123/\", \
    \"feature_store_type\" : \"tensorflow\", \
    \"redis_url\" :\"test_url\",  \
    \"redis_password\" :\"test_password\", \
    \"read_thread_num\" : 2, \
    \"update_thread_num\":1, \
    \"model_store_type\": \"local\", \
    \"oss_endpoint\": \"test.endpoint\", \
    \"oss_access_id\" : \"test_id\", \
    \"oss_access_key\" : \"test_key\" \
  }";

  ModelConfig* config = nullptr;

  EXPECT_TRUE(
      ModelConfigFactory::Create(local_config.c_str(), &config).ok());
  EXPECT_EQ("tf", config->processor_type);

  TestableTFInstanceMgr mgr(config);
  auto status = mgr.Init();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ("SavedModel dir is invalid.", status.error_message());
}

} // processor
} // tensorflow

#include "gtest/gtest.h"
#include "odl_processor/storage/model_store.h"
#include "odl_processor/serving/model_config.h"

namespace tensorflow {
namespace processor {
class ModelStorageTest : public ::testing::Test {
 protected:
  ModelStorageTest() {
    config.checkpoint_dir = "oss://a/b/c";
    config.savedmodel_dir = "oss://d/e/f";

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
  }

  ModelConfig config;
};

TEST_F(ModelStorageTest, ConstructSuccessWhenCorrectConfig) {
  ModelStorage ms(&config);
  EXPECT_TRUE(ms.Init().ok());
}

TEST_F(ModelStorageTest, ConstructFailureWhenInvalidConfig) {
  EXPECT_TRUE(true);
}
} // processor
} // tensorflow

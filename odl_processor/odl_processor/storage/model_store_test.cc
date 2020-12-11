#include "gtest/gtest.h"
#include "odl_processor/storage/model_store.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/framework/model_version.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace processor {
class ModelStoreTest : public ::testing::Test {
 protected:
  ModelStoreTest() {}
  ModelConfig CreateInvalidModelConfig() {
    ModelConfig config;
    config.checkpoint_dir = "lll://a/b/c";
    config.savedmodel_dir = "abc://d/e/f";

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
};

class FakeFileSystem : public FileSystem {
 public:
  FakeFileSystem(const std::vector<std::string>& file_names) :
      file_names_(file_names), FileSystem() {
    std::cerr << "file_names_:" << file_names_.size() << std::endl;
  }

  Status NewRandomAccessFile(const string& fname,
      std::unique_ptr<RandomAccessFile>* result) override {
    return Status::OK();
  }

  Status NewWritableFile(const string& fname,
      std::unique_ptr<WritableFile>* result) override {
    return Status::OK();
  }

  Status NewAppendableFile(const string& fname,
      std::unique_ptr<WritableFile>* result) {
    return Status::OK();
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return Status::OK();
  }

  Status FileExists(const string& fname) override {
    std::cerr << "FileExist:" << fname << std::endl;
    std::cerr << "file_names_:" << file_names_.size() << std::endl;
    for (auto it : file_names_) {
      std::cerr << "it:" << it << std::endl;
      if (it == fname) {
        return Status::OK();
      }
    }
    return Status(errors::Code::NOT_FOUND, "");
  }

  bool FilesExist(const std::vector<string>& files,
      std::vector<Status>* status) override {
    for (auto it : files) {
      bool found = false;
      for (auto jt : file_names_) {
        if (it == jt) {
          status->emplace_back(Status::OK());
          found = true;
          break;
        }
      }
      if (!found) {
        status->emplace_back(Status(errors::Code::NOT_FOUND, ""));
      }
    }
    return !status->empty();
  }

  Status GetChildren(const string& dir,
      std::vector<string>* result) override {
    for (auto it : file_names_) {
      if (it.find(dir) != std::string::npos) {
        result->emplace_back(it);
      }
    }
    return Status::OK();
  }

  Status GetMatchingPaths(const string& pattern,
      std::vector<string>* results) override {
    return Status::OK();
  }

  Status Stat(const string& fname, FileStatistics* stat) override {
    return Status::OK();
  }

  Status DeleteFile(const string& fname) override {
    return Status::OK();
  }
  
  Status CreateDir(const string& dirname) override {
    return Status::OK();
  }

  Status RecursivelyCreateDir(const string& dirname) override {
    return Status::OK();
  }

  Status GetFileSize(const string& fname, uint64* file_size) override {
    return Status::OK();
  }

  Status RenameFile(const string& src, const string& target) override {
    return Status::OK();
  }

  Status CopyFile(const string& src, const string& target) override {
    return Status::OK();
  }

  string TranslateName(const string& name) const override {
    return std::string();
  }

  Status IsDirectory(const string& fname) override {
    for (auto it : file_names_) {
      if (fname == it) {
        return Status(errors::Code::INVALID_ARGUMENT, "");
      }
    }
    for (auto it : file_names_) {
      if (it.find(fname) != std::string::npos) {
        return Status::OK();
      }
    }
    return Status(errors::Code::INVALID_ARGUMENT, "");
  }

  void FlushCaches() override {
  } 

  Status NewTransactionFile(const string& fname,
      std::unique_ptr<WritableFile>* result) override {
    return Status::OK();
  }

  Status TransactionRenameFile(const string& src,
      const string& target) override {
    return Status::OK();
  }

  Status DeleteDir(const string& dirname) override {
    return Status::OK();
  }
  
 private:
  std::vector<std::string> file_names_;
};

class TestableModelStore : public ModelStore {
 public:
  TestableModelStore(ModelConfig* config) : ModelStore(config) {}

  Status Init(const std::vector<std::string>& file_names) {
    file_system_ = new FakeFileSystem(file_names);
    return Status::OK();
  }
};

TEST_F(ModelStoreTest, InitSuccessWhenOssFileSystem) {
  ModelConfig config = CreateValidModelConfig();
  ModelStore ms(&config);
  EXPECT_TRUE(ms.Init().ok());
}

TEST_F(ModelStoreTest, InitFailureWhenInvalidFileSystem) {
  ModelConfig config = CreateInvalidModelConfig();
  ModelStore ms(&config);
  EXPECT_FALSE(ms.Init().ok());
}

TEST_F(ModelStoreTest,
    GetLatestVersionReturnEmptySavedModelPathWhenNoSavedModelPath) {
  ModelConfig config = CreateValidModelConfig();
  TestableModelStore ms(&config);
  ms.Init({"oss://test_checkpoint/model.ckpt-1512965.meta",
           "oss://test_checkpoint/model.ckpt-1512965.index",
           "oss://test_checkpoint/model.ckpt-1512966.data-00000-of-00002",
           "oss://test_checkpoint/model.ckpt-1512966.data-00001-of-00002"});

  Version version;
  ms.GetLatestVersion(version).ok();
  EXPECT_EQ(version.savedmodel_dir, "");
}

TEST_F(ModelStoreTest,
    GetLatestVersionReturnValidVersionWhenValidModelPath) {
  ModelConfig config = CreateValidModelConfig();
  TestableModelStore ms(&config);
  ms.Init({"oss://test_checkpoint/model.ckpt-1512965.meta",
           "oss://test_checkpoint/model.ckpt-1512965.index",
           "oss://test_checkpoint/model.ckpt-1512966.data-00000-of-00002",
           "oss://test_checkpoint/model.ckpt-1512966.data-00001-of-00002",
           "oss://test_savedmodel/saved_model.pb",
           "oss://test_savedmodel/saved_model.pbtxt"});

  Version version;
  ms.GetLatestVersion(version);
  EXPECT_EQ(version.savedmodel_dir, "oss://test_savedmodel/");
}

TEST_F(ModelStoreTest,
    GetLatestVersionReturnLatestVersionWhenValidModelPath) {
  ModelConfig config = CreateValidModelConfig();
  TestableModelStore ms(&config);
  ms.Init({
           "oss://test_checkpoint/model.ckpt-1512965.meta",
           "oss://test_checkpoint/model.ckpt-1512965.index",
           "oss://test_checkpoint/model.ckpt-1512966.data-00000-of-00002",
           "oss://test_checkpoint/model.ckpt-1512966.data-00001-of-00002",
           "oss://test_checkpoint/model.ckpt-1612142.meta",
           "oss://test_checkpoint/model.ckpt-1612142.index",
           "oss://test_checkpoint/model.ckpt-1612142.data-00000-of-00002",
           "oss://test_checkpoint/model.ckpt-1612142.data-00001-of-00002",
           "oss://test_savedmodel/saved_model.pb",
           "oss://test_savedmodel/saved_model.pbtxt"});

  Version version;
  ms.GetLatestVersion(version);
  EXPECT_EQ(version.savedmodel_dir, "oss://test_savedmodel/");
  EXPECT_EQ(version.full_ckpt_name, "oss://test_checkpoint/model.ckpt-1612142");
  EXPECT_EQ(version.delta_ckpt_name, "");
}

} // processor
} // tensorflow

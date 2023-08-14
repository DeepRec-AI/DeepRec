#include <thread>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/embedding_variable_test.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#endif //GOOGLE_CUDA

#include <time.h>
#include <sys/resource.h>
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#ifdef TENSORFLOW_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif

namespace tensorflow {
namespace embedding {
namespace {
const int THREADNUM = 16;
const int64 max = 2147483647;

struct ProcMemory {
  long size;      // total program size
  long resident;  // resident set size
  long share;     // shared pages
  long trs;       // text (code)
  long lrs;       // library
  long drs;       // data/stack
  long dt;        // dirty pages

  ProcMemory() : size(0), resident(0), share(0),
                 trs(0), lrs(0), drs(0), dt(0) {}
};

ProcMemory getProcMemory() {
  ProcMemory m;
  FILE* fp = fopen("/proc/self/statm", "r");
  if (fp == NULL) {
    LOG(ERROR) << "Fail to open /proc/self/statm.";
    return m;
  }

  if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
             &m.size, &m.resident, &m.share,
             &m.trs, &m.lrs, &m.drs, &m.dt) != 7) {
    fclose(fp);
    LOG(ERROR) << "Fail to fscanf /proc/self/statm.";
    return m;
  }
  fclose(fp);

  return m;
}

double getSize() {
  ProcMemory m = getProcMemory();
  return m.size;
}

double getResident() {
  ProcMemory m = getProcMemory();
  return m.resident;
}

string Prefix(const string& prefix) {
  return strings::StrCat(testing::TmpDir(), "/", prefix);
}

std::vector<string> AllTensorKeys(BundleReader* reader) {
  std::vector<string> ret;
  reader->Seek(kHeaderEntryKey);
  reader->Next();
  for (; reader->Valid(); reader->Next()) {
    //ret.push_back(reader->key().ToString());
    ret.push_back(std::string(reader->key()));
  }
  return ret;
}

TEST(EmbeddingVariableTest, TestEmptyEV) {
  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  {
    auto variable = CreateEmbeddingVar(value_size, value, 1);

    LOG(INFO) << "size:" << variable->Size();
    Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

    BundleWriter writer(Env::Default(), Prefix("foo"));
    embedding::ShrinkArgs shrink_args;
    shrink_args.global_step = 1;
    variable->Save("var/part_0", Prefix("foo"), &writer, shrink_args);
    TF_ASSERT_OK(writer.Finish());

    {
      BundleReader reader(Env::Default(), Prefix("foo"));
      TF_ASSERT_OK(reader.status());
      EXPECT_EQ(
          AllTensorKeys(&reader),
          std::vector<string>({"var/part_0-freqs", "var/part_0-freqs_filtered", "var/part_0-keys",
                               "var/part_0-keys_filtered", "var/part_0-partition_filter_offset",
                               "var/part_0-partition_offset", "var/part_0-values",
                               "var/part_0-versions", "var/part_0-versions_filtered"}));
      {
        string key = "var/part_0-keys";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read keys:" << val.DebugString();
      }
      {
        string key = "var/part_0-values";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0, value_size});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read values:" << val.DebugString();
      }
      {
        string key = "var/part_0-versions";
        EXPECT_TRUE(reader.Contains(key));
        // Tests for LookupDtypeAndShape().
        DataType dtype;
        TensorShape shape;
        TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
        // Tests for Lookup(), checking tensor contents.
        Tensor val(dtype, TensorShape{0});
        TF_ASSERT_OK(reader.Lookup(key, &val));
        LOG(INFO) << "read versions:" << val.DebugString();
      }
    }
  }
}

TEST(EmbeddingVariableTest, TestEVExportSmallLockless) {
  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  auto variable = CreateEmbeddingVar(value_size, value, 1, 0, 5);

  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  for (int64 i = 0; i < 5; i++) {
    void* value_ptr = nullptr;
    variable->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
    vflat(i) = 5.0;
  }

  LOG(INFO) << "size:" << variable->Size();

  BundleWriter writer(Env::Default(), Prefix("foo"));
  embedding::ShrinkArgs shrink_args;
  shrink_args.global_step = 1;
  variable->Save("var/part_0", Prefix("foo"), &writer, shrink_args);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-freqs", "var/part_0-freqs_filtered", "var/part_0-keys",
                               "var/part_0-keys_filtered", "var/part_0-partition_filter_offset",
                               "var/part_0-partition_offset", "var/part_0-values",
                               "var/part_0-versions", "var/part_0-versions_filtered"}));
    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5, value_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{5});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }
  }
}

TEST(EmbeddingVariableTest, TestEVExportLargeLockless) {

  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto variable = CreateEmbeddingVar(value_size, value, 1, 0, 5);

  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  int64 ev_size = 10048576;
  for (int64 i = 0; i < ev_size; i++) {
    void* value_ptr = nullptr;
    variable->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
  }

  LOG(INFO) << "size:" << variable->Size();

  BundleWriter writer(Env::Default(), Prefix("foo"));
  embedding::ShrinkArgs shrink_args;
  shrink_args.global_step = 1;
  variable->Save("var/part_0", Prefix("foo"), &writer, shrink_args);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-freqs", "var/part_0-freqs_filtered", "var/part_0-keys",
                               "var/part_0-keys_filtered", "var/part_0-partition_filter_offset",
                               "var/part_0-partition_offset", "var/part_0-values",
                               "var/part_0-versions", "var/part_0-versions_filtered"}));
    {
      string key = "var/part_0-keys";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read keys:" << val.DebugString();
    }
    {
      string key = "var/part_0-values";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size, value_size});
      LOG(INFO) << "read values:" << val.DebugString();
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read values:" << val.DebugString();
    }
    {
      string key = "var/part_0-versions";
      EXPECT_TRUE(reader.Contains(key));
      // Tests for LookupDtypeAndShape().
      DataType dtype;
      TensorShape shape;
      TF_ASSERT_OK(reader.LookupDtypeAndShape(key, &dtype, &shape));
      // Tests for Lookup(), checking tensor contents.
      Tensor val(dtype, TensorShape{ev_size});
      TF_ASSERT_OK(reader.Lookup(key, &val));
      LOG(INFO) << "read versions:" << val.DebugString();
    }
  }
}

void multi_insertion(EmbeddingVar<int64, float>* variable, int64 value_size){
  for (long j = 0; j < 5; j++) {
    void* value_ptr = nullptr;
    variable->LookupOrCreateKey(j, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
  }
}

TEST(EmbeddingVariableTest, TestMultiInsertion) {
  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto variable = CreateEmbeddingVar(value_size, value, 1);

  std::vector<std::thread> insert_threads(THREADNUM);
  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(multi_insertion,variable, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  ASSERT_EQ(variable->Size(), 5);
}

void InsertAndLookup(EmbeddingVar<int64, float>* variable,
                     int64 *keys, long ReadLoops, int value_size){
  for (long j = 0; j < ReadLoops; j++) {
    void* val = nullptr;
    void* val_1 = nullptr;
    bool is_filter = true;
    variable->LookupOrCreateKey(keys[j], &val, &is_filter, false);
    variable->LookupOrCreateKey(keys[j], &val_1, &is_filter, false);
    ASSERT_EQ(val, val_1);
  }
}

void MultiBloomFilter(EmbeddingVar<int64, float>* var, int value_size, int64 i) {
  for (long j = 0; j < 1; j++) {
    void* val = nullptr;
    bool is_filter = true;
    var->LookupOrCreateKey(i+1, &val, &is_filter, false);
  }
}

TEST(EmbeddingVariableTest, TestBloomFilter) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  std::vector<float> default_value =
      {0.0 ,1.0 ,2.0 ,3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  test::FillValues<float>(&value, default_value);

  auto var = CreateEmbeddingVar(value_size, value,
                                1, 3, 5, -1.0,
                                embedding::StorageType::DRAM,
                                {1024, 1024, 1024, 1024},
                                false, 10, 0.01);

  //float *val = (float *)malloc((value_size+1)*sizeof(float));
  void* val = nullptr;
  bool is_filter = true;
  var->LookupOrCreateKey(1, &val, &is_filter, false);
  var->LookupOrCreateKey(1, &val, &is_filter, false);
  var->LookupOrCreateKey(1, &val, &is_filter, false);
  var->LookupOrCreateKey(1, &val, &is_filter, false);
  var->LookupOrCreateKey(2, &val, &is_filter, false);
  
  std::vector<int64> keylist;
  std::vector<float *> valuelist;
  std::vector<int64> version_list;
  std::vector<int64> freq_list;

  ASSERT_EQ(var->Size(), 1);
}

TEST(EmbeddingVariableTest, TestBloomCounterInt64) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 
  auto var = CreateEmbeddingVar(value_size, value,
                                1, 3, 5, -1.0,
                                embedding::StorageType::DRAM,
                                {1024, 1024, 1024, 1024},
                                false, 10, 0.01, DT_UINT64);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilterPolicy<int64, float,
       EmbeddingVar<int64, float>>*>(filter);
  //(int64 *)var->GetBloomCounter(); 
  int64* counter = (int64*)bloom_filter->GetBloomCounter();

  for (auto it: hash_val1) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ(counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestBloomCounterInt32) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto var = CreateEmbeddingVar(value_size, value,
                                1, 3, 5, -1.0,
                                embedding::StorageType::DRAM,
                                {1024, 1024, 1024, 1024},
                                false, 10, 0.01, DT_UINT32);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilterPolicy<int64, float,
       EmbeddingVar<int64, float>>*>(filter);
  //(int64 *)var->GetBloomCounter(); 
  int32* counter = (int32*)bloom_filter->GetBloomCounter();

  for (auto it: hash_val1) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ(counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestBloomCounterInt16) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto var = CreateEmbeddingVar(value_size, value,
                                1, 3, 5, -1.0,
                                embedding::StorageType::DRAM,
                                {1024, 1024, 1024, 1024},
                                false, 10, 0.01, DT_UINT16);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  //int16* counter = (int16 *)var->GetBloomCounter(); 
  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilterPolicy<int64, float,
       EmbeddingVar<int64, float>>*>(filter);
  //(int64 *)var->GetBloomCounter(); 
  int16* counter = (int16*)bloom_filter->GetBloomCounter();

  for (auto it: hash_val1) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ(counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ(counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestBloomCounterInt8) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  auto var = CreateEmbeddingVar(value_size, value,
                                1, 3, 5, -1.0,
                                embedding::StorageType::DRAM,
                                {1024, 1024, 1024, 1024},
                                false, 10, 0.01, DT_UINT8);

  float *val = (float *)malloc((value_size+1)*sizeof(float));

  std::vector<int64> hash_val1= {17, 7, 48, 89, 9, 20, 56};
  std::vector<int64> hash_val2= {58, 14, 10, 90, 28, 14, 67};
  std::vector<int64> hash_val3= {64, 63, 9, 77, 7, 38, 11};
  std::vector<int64> hash_val4= {39, 10, 79, 28, 58, 55, 60};

  std::map<int64, int> tab;
  for (auto it: hash_val1)
    tab.insert(std::pair<int64,int>(it, 1));
  for (auto it: hash_val2) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val3) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }
  for (auto it: hash_val4) {
    if (tab.find(it) != tab.end())
      tab[it]++;
    else
      tab.insert(std::pair<int64,int>(it, 1));
  }

  std::vector<std::thread> insert_threads(4);
  for (size_t i = 0 ; i < 4; i++) {
    insert_threads[i] = std::thread(MultiBloomFilter, var, value_size, i);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  auto filter = var->GetFilter();
  auto bloom_filter = static_cast<BloomFilterPolicy<int64, float,
       EmbeddingVar<int64, float>>*>(filter);
  int8* counter = (int8*)bloom_filter->GetBloomCounter();
  //(int64 *)var->GetBloomCounter(); 
  //int8* counter = (int8 *)var->GetBloomCounter(); 

  for (auto it: hash_val1) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
  for (auto it: hash_val2) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
  for (auto it: hash_val3) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
  for (auto it: hash_val4) {
    ASSERT_EQ((int)counter[it], tab[it]);
  }
}

TEST(EmbeddingVariableTest, TestInsertAndLookup) {
  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10));
  auto variable = CreateEmbeddingVar(value_size, value, 1);

  int64 InsertLoops = 1000;
  bool* flag = (bool *)malloc(sizeof(bool)*max);
  srand((unsigned)time(NULL));
  int64 *keys = (int64 *)malloc(sizeof(int64)*InsertLoops);

  for (long i = 0; i < max; i++) {
    flag[i] = 0;
  }

  int index = 0;
  while (index < InsertLoops) {
    long j = rand() % max;
    if (flag[j] == 1) // the number is already set as a key
      continue;
    else { // the number is not selected as a key
      keys[index] = j;
      index++;
      flag[j] = 1;
    }
  }
  free(flag);
  std::vector<std::thread> insert_threads(THREADNUM);
  for (size_t i = 0 ; i < THREADNUM; i++) {
    insert_threads[i] = std::thread(InsertAndLookup,
        variable, &keys[i*InsertLoops/THREADNUM],
        InsertLoops/THREADNUM, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }
}

void MultiFilter(EmbeddingVar<int64, float>* variable, int value_size) {
  bool is_filter = true;
  void* val;
  variable->LookupOrCreateKey(20, &val, &is_filter, false);
}

TEST(EmbeddingVariableTest, TestFeatureFilterParallel) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 
  auto var = CreateEmbeddingVar(value_size, value, 1, 7, 5);

  float *val = (float *)malloc((value_size+1)*sizeof(float));
  int thread_num = 5;
  std::vector<std::thread> insert_threads(thread_num);
  for (size_t i = 0 ; i < thread_num; i++) {
    insert_threads[i] = std::thread(MultiFilter, var, value_size);
  }
  for (auto &t : insert_threads) {
    t.join();
  }

  void* value_ptr = nullptr;
  var->LookupOrCreateKey(20, &value_ptr);
  ASSERT_EQ(var->GetFreq(20), thread_num);
}

EmbeddingVar<int64, float>* InitEV_Lockless(int64 value_size) {
  Tensor value(DT_INT64, TensorShape({value_size}));
  test::FillValues<int64>(&value, std::vector<int64>(value_size, 10));
  auto variable = CreateEmbeddingVar(value_size, value, 1);

  return variable;
}

void MultiLookup(EmbeddingVar<int64, float>* variable,
    int64 InsertLoop, int thread_num, int i) {
  for (int64 j = i * InsertLoop/thread_num;
      j < (i+1)*InsertLoop/thread_num; j++) {
    void* value_ptr = nullptr;
    variable->LookupOrCreateKey(j, &value_ptr);
  }
}

void BM_MULTIREAD_LOCKLESS(int iters, int thread_num) {
  testing::StopTiming();
  testing::UseRealTime();

  int64 value_size = 128;
  auto variable = InitEV_Lockless(value_size);
  int64 InsertLoop =  1000000;

  float* fill_v = (float*)malloc(value_size * sizeof(float));

  for (int64 i = 0; i < InsertLoop; i++){
    void* value_ptr = nullptr;
    variable->LookupOrCreateKey(i, &value_ptr);
    typename TTypes<float>::Flat vflat = variable->flat(value_ptr);
  }

  testing::StartTiming();
  while(iters--){
    std::vector<std::thread> insert_threads(thread_num);
    for (size_t i = 0 ; i < thread_num; i++) {
      insert_threads[i] = std::thread(MultiLookup,
          variable, InsertLoop, thread_num, i);
    }
    for (auto &t : insert_threads) {
      t.join();
    }
  }

}


TEST(EmbeddingVariableTest, TestAllocate) {
  int value_len = 8;
  double t0 = getResident()*getpagesize()/1024.0/1024.0;
  double t1 = 0;
  LOG(INFO) << "memory t0: " << t0;
  for (int64 i = 0; i < 1000; ++i) {
    float* tensor_val = TypedAllocator::Allocate<float>(
        ev_allocator(), value_len, AllocationAttributes());
    t1 = getResident()*getpagesize()/1024.0/1024.0;
    memset(tensor_val, 0, sizeof(float) * value_len);
  }
  double t2 = getResident()*getpagesize()/1024.0/1024.0;
  LOG(INFO) << "memory t1-t0: " << t1-t0;
  LOG(INFO) << "memory t2-t1: " << t2-t1;
  LOG(INFO) << "memory t2-t0: " << t2-t0;
}

TEST(EmbeddingVariableTest, TestEVStorageType_DRAM) {
  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto variable = CreateEmbeddingVar(value_size, value, 1);

  int64 ev_size = 100;
  for (int64 i = 0; i < ev_size; i++) {
    void* val = nullptr;
    bool is_filter = true;
    variable->LookupOrCreateKey(i, &val, &is_filter, false);
  }

  LOG(INFO) << "size:" << variable->Size();
}

void t1(KVInterface<int64, float>* hashmap) {
  for (int i = 0; i< 100; ++i) {
    hashmap->Insert(i, nullptr);
  }
}

TEST(EmbeddingVariableTest, TestRemoveLockless) {
  auto feat_desc = new embedding::FeatureDescriptor<float>(
      1, 1, ev_allocator(), embedding::StorageType::DRAM,
      false, false, {false, 0});
  KVInterface<int64, float>* hashmap =
      new LocklessHashMap<int64, float>(feat_desc);
  feat_desc->InitSlotInfo(0, 100, {nullptr, 1});
  ASSERT_EQ(hashmap->Size(), 0);
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  auto t = std::thread(t1, hashmap);
  t.join();
  LOG(INFO) << "hashmap size: " << hashmap->Size();
  ASSERT_EQ(hashmap->Size(), 100);
  TF_CHECK_OK(hashmap->Remove(1));
  TF_CHECK_OK(hashmap->Remove(2));
  ASSERT_EQ(hashmap->Size(), 98);
  LOG(INFO) << "2 size:" << hashmap->Size();
}

TEST(EmbeddingVariableTest, TestLRUCachePrefetch) {
  BatchCache<int64>* cache = new LRUCache<int64>();
  int num_ids = 5;
  std::vector<int64> prefetch_ids;
  int index = 0;
  int64 true_evict_size;
  int64* evict_ids = new int64[num_ids];
  std::vector<int64> access_seq;
  for(int i = 1; i <= num_ids; i++) {
    for (int j = 0; j < i; j++) {
      prefetch_ids.emplace_back(i);
    }
  }
  cache->add_to_prefetch_list(prefetch_ids.data(), prefetch_ids.size());
  ASSERT_EQ(cache->size(), 0);
  true_evict_size = cache->get_evic_ids(evict_ids, num_ids);
  ASSERT_EQ(true_evict_size, 0);
  for (int i = 1; i <= 2; i++) {
    for (int j = 0; j < i; j++) {
      access_seq.emplace_back(i);
    }
  }
  cache->add_to_cache(access_seq.data(), access_seq.size());
  ASSERT_EQ(cache->size(), 2);
  true_evict_size = cache->get_evic_ids(evict_ids, num_ids);
  ASSERT_EQ(true_evict_size, 2);
  access_seq.clear();
  for (int i = 5; i >= 3; i--) {
    for (int j = 0; j < i; j++) {
      access_seq.emplace_back(i);
    }
  }
  cache->add_to_cache(access_seq.data(), access_seq.size());
  ASSERT_EQ(cache->size(), 3);
  true_evict_size = cache->get_evic_ids(evict_ids, 2);
  ASSERT_EQ(evict_ids[0], 5);
  ASSERT_EQ(evict_ids[1], 4);
  ASSERT_EQ(cache->size(), 1);

  delete cache;
  delete[] evict_ids;
}

TEST(EmbeddingVariableTest, TestLRUCache) {
  BatchCache<int64>* cache = new LRUCache<int64>();
  int num_ids = 30;
  int num_access = 100;
  int num_evict = 50;
  int64 ids[num_access] = {0};
  int64 evict_ids[num_evict] = {0};
  for (int i = 0; i < num_access; i++){
    ids[i] = i % num_ids;
  }
  cache->update(ids, num_access);
  int64 size = cache->get_evic_ids(evict_ids, num_evict);
  ASSERT_EQ(size, num_ids);
  ASSERT_EQ(cache->size(), 0);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(evict_ids[i], (num_access % num_ids + i) % num_ids);
  }
}

TEST(EmbeddingVariableTest, TestLRUCacheGetCachedIds) {
  BatchCache<int64>* cache = new LRUCache<int64>();
  int num_ids = 30;
  int num_access = 100;
  int num_evict = 15;
  int num_cache = 20;
  int64 ids[num_access] = {0};
  int64 evict_ids[num_evict] = {0};
  for (int i = 0; i < num_access; i++){
    ids[i] = i % num_ids;
  }
  cache->update(ids, num_access);
  ASSERT_EQ(cache->size(), num_ids);
  int64* cached_ids = new int64[num_cache];
  int64* cached_freqs = new int64[num_cache];
  int64 true_size = 
      cache->get_cached_ids(cached_ids, num_cache, nullptr, cached_freqs);
  ASSERT_EQ(true_size, 20);
  cache->get_evic_ids(evict_ids, num_evict);
  ASSERT_EQ(cache->size(), 15);
  true_size =
      cache->get_cached_ids(cached_ids, num_cache, nullptr, cached_freqs);
  ASSERT_EQ(true_size, 15);
  delete cache;
  delete[] cached_ids;
  delete[] cached_freqs;
}

TEST(EmbeddingVariableTest, TestLFUCacheGetCachedIds) {
  BatchCache<int64>* cache = new LFUCache<int64>();
  int num_ids = 30;
  int num_access = 100;
  int num_evict = 15;
  int num_cache = 20;
  int64 ids[num_access] = {0};
  int64 evict_ids[num_evict] = {0};
  for (int i = 0; i < num_access; i++){
    ids[i] = i % num_ids;
  }
  cache->update(ids, num_access);
  ASSERT_EQ(cache->size(), num_ids);
  int64* cached_ids = new int64[num_cache];
  int64* cached_freqs = new int64[num_cache];
  int64 true_size = 
      cache->get_cached_ids(cached_ids, num_cache, nullptr, cached_freqs);
  ASSERT_EQ(true_size, 20);
  cache->get_evic_ids(evict_ids, num_evict);
  ASSERT_EQ(cache->size(), 15);
  true_size =
      cache->get_cached_ids(cached_ids, num_cache, nullptr, cached_freqs);
  ASSERT_EQ(true_size, 15);
  delete cache;
  delete[] cached_ids;
  delete[] cached_freqs;
}

TEST(EmbeddingVariableTest, TestLFUCachePrefetch) {
  BatchCache<int64>* cache = new LFUCache<int64>();
  int num_ids = 5;
  std::vector<int64> prefetch_ids;
  int index = 0;
  int64 true_evict_size;
  int64* evict_ids = new int64[num_ids];
  std::vector<int64> access_seq;
  for(int i = 1; i <= num_ids; i++) {
    for (int j = 0; j < i; j++) {
      prefetch_ids.emplace_back(i);
    }
  }
  cache->add_to_prefetch_list(prefetch_ids.data(), prefetch_ids.size());
  ASSERT_EQ(cache->size(), 0);
  true_evict_size = cache->get_evic_ids(evict_ids, num_ids);
  ASSERT_EQ(true_evict_size, 0);
  for (int i = 1; i <= 2; i++) {
    for (int j = 0; j < i; j++) {
      access_seq.emplace_back(i);
    }
  }
  cache->add_to_cache(access_seq.data(), access_seq.size());
  ASSERT_EQ(cache->size(), 2);
  true_evict_size = cache->get_evic_ids(evict_ids, num_ids);
  ASSERT_EQ(true_evict_size, 2);
  access_seq.clear();
  for (int i = 5; i >= 3; i--) {
    for (int j = 0; j < i; j++) {
      access_seq.emplace_back(i);
    }
  }
  cache->add_to_cache(access_seq.data(), access_seq.size());
  ASSERT_EQ(cache->size(), 3);
  true_evict_size = cache->get_evic_ids(evict_ids, 2);
  ASSERT_EQ(evict_ids[0], 3);
  ASSERT_EQ(evict_ids[1], 4);
  ASSERT_EQ(cache->size(), 1);

  delete cache;
  delete[] evict_ids;
}


TEST(EmbeddingVariableTest, TestLFUCache) {
  BatchCache<int64>* cache = new LFUCache<int64>();
  int num_ids = 30;
  int num_access = 100;
  int num_evict = 50;
  int64 ids[num_access] = {0};
  int64 evict_ids[num_evict] = {0};
  for (int i = 0; i < num_access; i++){
    ids[i] = i % num_ids;
  }
  cache->update(ids, num_access);
  int64 size = cache->get_evic_ids(evict_ids, num_evict);
  ASSERT_EQ(size, num_ids);
  ASSERT_EQ(cache->size(), 0);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(evict_ids[i], (num_access % num_ids + i) % num_ids);
  }
}

const int total_size = 1024 * 8;
const int th_num = 1;
const int malloc_size = total_size / th_num;

void malloc_use_allocator(Allocator* allocator){
  timespec start;
  timespec end;
  float* first = (float *)allocator->AllocateRaw(0, sizeof(float));

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < malloc_size; ++i) {
    int ev_list_size = 32;
    float* ptr_ = (float *)allocator->AllocateRaw(
        0, sizeof(float) * ev_list_size);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "cost time: "
            << ((double)(end.tv_sec - start.tv_sec) *
                1000000000 + end.tv_nsec - start.tv_nsec) / 1000000
            << "ms";
}

TEST(EmbeddingVariableTest, TestEVMalloc) {
  std::thread th_arr[th_num];
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i] = std::thread(malloc_use_allocator, ev_allocator());
  }
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i].join();
  }
}

TEST(EmbeddingVariableTest, TestCPUMalloc) {
  std::thread th_arr[th_num];
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i] = std::thread(malloc_use_allocator, cpu_allocator());
  }
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i].join();
  }
}

#if GOOGLE_CUDA
TEST(EmbeddingVariableTest, TestGPUMalloc) {
  SessionOptions sops;
  std::unique_ptr<Device> device =
    DeviceFactory::NewDevice(DEVICE_GPU, sops, "/job:a/replica:0/task:0");
  Allocator* gpu_allocator = GPUProcessState::singleton()->GetGPUAllocator(
        GPUOptions(), TfGpuId(0), 1 << 26);

  std::thread th_arr[th_num];
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i] = std::thread(malloc_use_allocator, gpu_allocator);
  }
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i].join();
  }
}

TEST(EmbeddingVariableTest, TestCPUGPUMalloc) {
  SessionOptions sops;
  std::unique_ptr<Device> device =
    DeviceFactory::NewDevice(DEVICE_GPU, sops, "/job:a/replica:0/task:0");

  auto gpu_allocator = GPUProcessState::singleton()->GetGPUAllocator(
        GPUOptions(), TfGpuId(0), 1 << 26);
  auto mem_pool = new EmbeddingMemoryPool<float>(gpu_allocator, 256, 1024);
  float* ptr_1 = mem_pool->Allocate();
  float* ptr_2 = mem_pool->Allocate();
  std::vector<void*> value_ptrs;
  value_ptrs.emplace_back(ptr_1);
  mem_pool->Deallocate(value_ptrs);
  value_ptrs.clear();
  value_ptrs.emplace_back(ptr_2);
  mem_pool->Deallocate(value_ptrs);
  float* ptr_3 = mem_pool->Allocate();
  ASSERT_EQ(ptr_1, ptr_3);
  delete mem_pool;
}
#endif //GOOGLE_CUDA

void malloc_free_use_allocator(Allocator* allocator){
  timespec start;
  timespec end;
  std::vector<float*> ptrs;
  float* first = (float *)allocator->AllocateRaw(0, sizeof(float));

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < malloc_size; ++i) {
    int ev_list_size = 32;
    float* ptr_ = (float *)allocator->AllocateRaw(
        0, sizeof(float) * ev_list_size);
    ptrs.push_back(ptr_);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "first time: "
            << ((double)(end.tv_sec - start.tv_sec) *
                1000000000 + end.tv_nsec - start.tv_nsec) / 1000000
            << "ms";

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (auto iter = ptrs.begin();iter != ptrs.end();iter++) {
    allocator->DeallocateRaw(*iter);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "free time: "
            << ((double)(end.tv_sec - start.tv_sec) *
                1000000000 + end.tv_nsec - start.tv_nsec) / 1000000
            << "ms";

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < malloc_size; ++i) {
    int ev_list_size = 32;
    float* ptr_ = (float *)allocator->AllocateRaw(
        0, sizeof(float) * ev_list_size);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  LOG(INFO) << "second time: "
            << ((double)(end.tv_sec - start.tv_sec) *
                1000000000 + end.tv_nsec - start.tv_nsec) / 1000000
            << "ms";
}

TEST(EmbeddingVariableTest, TestEVMallocFree) {
  std::thread th_arr[th_num];
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i] = std::thread(
        malloc_free_use_allocator, ev_allocator());
  }
  for (unsigned int i = 0; i < th_num; ++i) {
    th_arr[i].join();
  }
}

void SingleCommit(KVInterface<int64, float>* hashmap,
    std::vector<int64> keys, int bias) {
  std::vector<void*> value_ptrs;
  for (int64 i = 0; i < keys.size(); ++i) {
    void* tmp = cpu_allocator()->AllocateRaw(0, 124 * sizeof(float) + 16);
    for (int j = 0; j < 124; j++) {
      ((float*)tmp)[j] = keys[i] + bias;
    }
    value_ptrs.push_back(tmp);
  }
  ASSERT_EQ(keys.size(), value_ptrs.size());
  uint64 start = Env::Default()->NowNanos();
  for (int64 i = 0; i < keys.size(); i++) {
    hashmap->Commit(keys[i], value_ptrs[i]);
  }
  uint64 end = Env::Default()->NowNanos();
  uint64 result_cost = end - start;
}

void TestCompaction() {
  std::string temp_dir = testing::TmpDir();
  auto feat_desc = new embedding::FeatureDescriptor<float>(
      1, 1, ev_allocator(), embedding::StorageType::DRAM_SSDHASH,
      true, true, {false, 0});
  auto hashmap = new SSDHashKV<int64, float>(
      temp_dir, feat_desc);
  feat_desc->InitSlotInfo(0, 124, {nullptr, 1});
  hashmap->Init();
  ASSERT_EQ(hashmap->Size(), 0);
  std::vector<int64> ids;
  for (int i = 0; i < 262144; i++) {
    ids.emplace_back(i);
  }
  auto t1 = std::thread(SingleCommit, hashmap, ids, 3);
  t1.join();
  ids.clear();
  for (int i = 0; i < 131073; i++) {
    ids.emplace_back(i);
  }
  t1 = std::thread(SingleCommit, hashmap, ids, 1);
  t1.join();
  ids.clear();
  sleep(1);
  void* val = nullptr;
  for (int i = 131073; i < 262144; i++) {
    hashmap->Lookup(i, &val);
    float* v = (float*)val;
    for (int j = 0; j < 124; j++){
      ASSERT_EQ(v[j], i+3);
    }
  }
  for (int i = 131073; i < 262144; i++) {
    ids.emplace_back(i);
  }
  t1 = std::thread(SingleCommit, hashmap, ids, 2);
  t1.join();
  ids.clear();
  ids.emplace_back(262155);
  t1 = std::thread(SingleCommit, hashmap, ids, 0);
  t1.join();
  sleep(1);
  for (int i = 0; i < 131073; i++) {
    hashmap->Lookup(i, &val);
    float* v = (float*)val;
    for (int j = 0; j < 124; j++){
      ASSERT_EQ(v[j], i + 1);
    }
  }
  for (int i = 131073; i < 262144; i++) {
    hashmap->Lookup(i, &val);
    float* v = (float*)val;
    for (int j = 0; j < 124; j++){
      ASSERT_EQ(v[j], i + 2);
    }
  }
  delete hashmap;
}

TEST(KVInterfaceTest, TestSSDKVAsyncCompaction) {
  setenv("TF_SSDHASH_ASYNC_COMPACTION", "true", 1);
  TestCompaction();
}

TEST(KVInterfaceTest, TestSSDKVSyncCompaction) {
  setenv("TF_SSDHASH_ASYNC_COMPACTION", "false", 1);
  TestCompaction();
}

void TestReadEmbFile() {
  auto feat_desc = new embedding::FeatureDescriptor<float>(
      1, 1, ev_allocator(), embedding::StorageType::DRAM_SSDHASH,
      true, true, {false, 0});
  std::string temp_dir = testing::TmpDir();
  auto hashmap = new SSDHashKV<int64, float>(
      temp_dir, feat_desc);
  feat_desc->InitSlotInfo(0, 124, {nullptr, 1});
  hashmap->Init();
  ASSERT_EQ(hashmap->Size(), 0);
  std::vector<int64> ids;
  for (int i = 0; i < 262145; i++) {
    ids.emplace_back(i);
  }
  SingleCommit(hashmap, ids, 3);
  sleep(1);
  ids.clear();
  void* val = nullptr;
  for (int i = 0; i < 262144; i++) {
    hashmap->Lookup(i, &val);
    float* v = (float*)val;
    for (int j = 0; j < 124; j++){
      ASSERT_EQ(v[j], i+3);
    }
  }
  delete hashmap;
}

TEST(KVInterfaceTest, TestMmapMadviseFile) {
  setenv("TF_SSDHASH_IO_SCHEME", "mmap_and_madvise", 1);
  TestReadEmbFile();
}

TEST(KVInterfaceTest, TestMmapFile) {
  std::string temp_dir = testing::TmpDir();
  setenv("TF_SSDHASH_IO_SCHEME", "mmap", 1);
  TestReadEmbFile();
}

TEST(KVInterfaceTest, TestDirectIoFile) {
  std::string temp_dir = testing::TmpDir();
  setenv("TF_SSDHASH_IO_SCHEME", "directio", 1);
  TestReadEmbFile();
}


void InsertKey(EmbeddingVar<int64, float>* variable, int value_size) {
  float *val = (float *)malloc((value_size+1)*sizeof(float));
  for (int64 i = 0; i < 100000000; i++) {
    void* val = nullptr;
    bool is_filter = true;
    variable->LookupOrCreateKey(20, &val, &is_filter, false);
  }
}

void RemoveKey(EmbeddingVar<int64, float>* variable) {
  for (int64 i = 0; i < 10; i++) {
    sleep(1);
    variable->storage()->Remove(20);
  }
}

TEST(EmbeddingVariableTest, TestLookupRemoveConcurrency) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  auto var = CreateEmbeddingVar(value_size, value, 1);
   int thread_num = 5;
   std::vector<std::thread> insert_threads(thread_num);
   for (size_t i = 0 ; i < thread_num - 1; i++) {
     insert_threads[i] = std::thread(InsertKey, var, value_size);
   }
   insert_threads[thread_num - 1] = std::thread(RemoveKey, var);
   for (auto &t : insert_threads) {
     t.join();
   }
}

TEST(EmbeddingVariableTest, TestInsertAndGetSnapshot) {
  int value_size = 10;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 10.0));
  auto var = CreateEmbeddingVar(value_size, value, 1);
  float* set_value = (float*)malloc(value_size * sizeof(float));
  //Insertion
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < value_size; j++) {
      set_value[j] = i + j;
    }
    var->Insert(i, set_value);
  }
  free(set_value);
  //GetSnapshot
  std::vector<int64> key_list;
  std::vector<float*> value_ptr_list;
  std::vector<int64> version_list;
  std::vector<int64> freq_list;
  var->GetSnapshot(&key_list, &value_ptr_list,
                  &version_list, &freq_list);
  for (int i = 0; i < key_list.size(); i++) {
    ASSERT_EQ(key_list[i], i);
    for (int j = 0; j < value_size; j++) {
      ASSERT_EQ(value_ptr_list[i][j], i + j);
    }
  }
}

} // namespace
} // namespace embedding
} // namespace tensorflow

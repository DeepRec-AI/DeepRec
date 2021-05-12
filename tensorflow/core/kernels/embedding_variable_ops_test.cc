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
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

#include <sys/resource.h>
#include "tensorflow/core/framework/hashmap.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#ifdef TENSORFLOW_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif

namespace tensorflow {
namespace {

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


TEST(TensorBundleTest, TestEVShrink) {

  int64 value_size = 64;
  int64 insert_num = 30;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  HashMap<int64, float>* emb_var = new HashMap<int64, float>(
      new DenseHashMap<int64, float>(), cpu_allocator());
  emb_var ->Init(value);


  LOG(INFO) << "size:" << emb_var->Size();


  for (int64 i=0; i < insert_num; ++i) {
    emb_var->LookupOrCreate(i, fill_v);
  }

  int size = emb_var->Size();
  emb_var->Shrink(5, insert_num);
  LOG(INFO) << "Before shrink size:" << size;
  LOG(INFO) << "After shrink size:" << emb_var->Size();

}


TEST(EmbeddingVariableTest, TestEmptyEV) {
  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));


  {
    EmbeddingVar<int64, float>* variable 
              = new EmbeddingVar<int64, float>("EmbeddingVar",
                  new HashMap<int64, float>(
                    new DenseHashMap<int64, float>(), cpu_allocator()),
                  1);
    variable->Init(value); 

    LOG(INFO) << "size:" << variable->hashmap()->Size();


    BundleWriter writer(Env::Default(), Prefix("foo"));
    DumpEmbeddingValues(variable, "var/part_0", &writer);
    TF_ASSERT_OK(writer.Finish()); 

    {
      BundleReader reader(Env::Default(), Prefix("foo"));
      TF_ASSERT_OK(reader.status());
      EXPECT_EQ(
          AllTensorKeys(&reader),
          std::vector<string>({"var/part_0-keys", "var/part_0-values", "var/part_0-versions"}));
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

TEST(EmbeddingVariableTest, TestEVExportSmall) {

  int64 value_size = 8;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));

  EmbeddingVar<int64, float>* variable 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DenseHashMap<int64, float>(), cpu_allocator()),
        1);
  variable->Init(value); 
  
  for (int64 i = 0; i < 5; i++) {
    typename TTypes<float>::Flat vflat = variable->hashmap()->flat(i, i);
    vflat(i) = 5.0;
  }

  LOG(INFO) << "size:" << variable->hashmap()->Size();


  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer);
  TF_ASSERT_OK(writer.Finish()); 

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-keys", "var/part_0-values", "var/part_0-versions"}));
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

TEST(EmbeddingVariableTest, TestEVExportLarge) {

  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  EmbeddingVar<int64, float>* variable 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DenseHashMap<int64, float>(), cpu_allocator()),
        0);
  variable->Init(value); 

  int64 ev_size = 10048576;
  for (int64 i = 0; i < ev_size; i++) {
    variable->hashmap()->LookupOrCreate(i, fill_v);
  }
 
  LOG(INFO) << "size:" << variable->hashmap()->Size();

  BundleWriter writer(Env::Default(), Prefix("foo"));
  DumpEmbeddingValues(variable, "var/part_0", &writer);
  TF_ASSERT_OK(writer.Finish());

  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"var/part_0-keys", "var/part_0-values", "var/part_0-versions"}));

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

TEST(EmbeddingVariableTest, TestColdDataStorage) {

  int64 value_size = 4096;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float)); 

  EmbeddingVar<int64, float>* variable 
    = new EmbeddingVar<int64, float>("EmbeddingVar",
        new HashMap<int64, float>(
          new DenseHashMap<int64, float>(),
          cpu_allocator(),
          true));

  variable->Init(value); 
  LOG(INFO) << "begin write " << variable->hashmap()->HybridSize(); 

  float* db_values = (float*)malloc(value_size * sizeof(float)); 
  for (int i = 0; i < 1024; i++) {
    variable->hashmap()->HybridInsert(i, db_values);
  }


  LOG(INFO) << "after write " << variable->hashmap()->HybridSize(); 
  srand((unsigned) time(NULL));
  float* gather_values = (float*)malloc(value_size * sizeof(float)); 
  for (int64 i = 0; i < 1024; i++) {
      variable->hashmap()->LookupOrCreateHybrid(i, gather_values, fill_v);
  }

}
    
} // namespace
} // namespace tensorflow

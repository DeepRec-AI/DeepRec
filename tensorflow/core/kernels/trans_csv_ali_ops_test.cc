// Copyright (c) 2017, Alibaba Inc.
// All right reserved.
//
// Author: Cao Zongyan <zongyan.cao@alibaba-inc.com>
// Created: 2017/09/20
//
// Description
//     Unit test cases for TransCSVxxxOp.
//

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

class TransCsvID2SparseTest : public OpsTestBase {
 protected:

  void CreateOp(DataType dtype, bool isset, string delim) {
    TF_ASSERT_OK(NodeDefBuilder("op", "TransCsvID2Sparse")
                     .Input(FakeInput(DT_STRING))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(dtype))
                     .Attr("id_as_value", isset)
                     .Attr("field_delim", delim)
                     .Finalize(node_def()));
  }
};

TEST_F(TransCsvID2SparseTest, NormalProcess_int64) {
  CreateOp(DT_INT64, true, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2,10", "7", "8,0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {-1048575});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices(allocator(), DT_INT64, {5, 2});
  test::FillValues<int64>(&expected_indices, {0, 2, 0, 10, 1, 7, 2, 0, 2, 8});
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_INT64, {5});
  test::FillValues<int64>(&expected_values, {2, 10, 7, 0, 8});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {2});
  test::FillValues<int64>(&expected_shape, {3, 11});
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

TEST_F(TransCsvID2SparseTest, NormalProcess_int32) {
  CreateOp(DT_INT32, false, "~");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2~10~7", "", "8~0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {15});
  // default value
  AddInputFromArray<int32>(TensorShape({}), {-1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices(allocator(), DT_INT64, {5, 2});
  test::FillValues<int64>(&expected_indices, {0, 2, 0, 7, 0, 10, 2, 0, 2, 8});
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_INT32, {5});
  test::FillValues<int32>(&expected_values, {-1, -1, -1, -1, -1});
  test::ExpectTensorEqual<int32>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {2});
  test::FillValues<int64>(&expected_shape, {3, 15});
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

TEST_F(TransCsvID2SparseTest, NormalProcess_float) {
  CreateOp(DT_FLOAT, false, "~");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({4}), {"2~  10", "7~ 8 ~ 0", "", ""});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {11});
  // default value
  AddInputFromArray<float>(TensorShape({}), {0.1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices(allocator(), DT_INT64, {5, 2});
  test::FillValues<int64>(&expected_indices, {0, 2, 0, 10, 1, 0, 1, 7, 1, 8});
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_FLOAT, {5});
  test::FillValues<float>(&expected_values, {0.1, 0.1, 0.1, 0.1, 0.1});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {2});
  test::FillValues<int64>(&expected_shape, {4, 11});
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

TEST_F(TransCsvID2SparseTest, DelimIsSpace) {
  CreateOp(DT_INT64, true, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2  10  ", "7", "  8   0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {-1048575});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices(allocator(), DT_INT64, {5, 2});
  test::FillValues<int64>(&expected_indices, {0, 2, 0, 10, 1, 7, 2, 0, 2, 8});
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_INT64, {5});
  test::FillValues<int64>(&expected_values, {2, 10, 7, 0, 8});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {2});
  test::FillValues<int64>(&expected_shape, {3, 11});
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

TEST_F(TransCsvID2SparseTest, InvalidDelim) {
  CreateOp(DT_INT64, true, "1");
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, InitOp().code());
}

TEST_F(TransCsvID2SparseTest, InvalidIndex) {
  CreateOp(DT_INT64, true, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2,10", "abc", "8,0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvID2SparseTest, MissingIndex) {
  CreateOp(DT_INT64, true, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2,10", " ,5", "8,0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvID2SparseTest, MaxIdNotBigEnough) {
  CreateOp(DT_INT64, true, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2,10", "7", "8,0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {8});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}


class TransCsvID2DenseTest : public OpsTestBase {
 protected:

  void CreateOp(DataType dtype, bool isset, string delim) {
    TF_ASSERT_OK(NodeDefBuilder("op", "TransCsvID2Dense")
                     .Input(FakeInput(DT_STRING))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(dtype))
                     .Attr("id_as_value", isset)
                     .Attr("field_delim", delim)
                     .Finalize(node_def()));
  }
};

TEST_F(TransCsvID2DenseTest, NormalProcess_int64) {
  CreateOp(DT_INT64, true, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2,10", "7", "8,0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {11});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_INT64, {3, 11});
  test::FillValues<int64>(&expected_values, {
                          0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 10,
                          0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvID2DenseTest, NormalProcess_int32) {
  CreateOp(DT_INT32, false, "~");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2~10~7", "", "8~0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {15});
  // default value
  AddInputFromArray<int32>(TensorShape({}), {-1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_INT32, {3, 15});
  test::FillValues<int32>(&expected_values, {
                          0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0});
  test::ExpectTensorEqual<int32>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvID2DenseTest, NormalProcess_float) {
  CreateOp(DT_FLOAT, false, "~");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({4}), {"2~  10", "7~ 8 ~ 0", "", ""});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {-1048575});
  // default value
  AddInputFromArray<float>(TensorShape({}), {0.1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_FLOAT, {4, 11});
  test::FillValues<float>(&expected_values, {
                          0., 0., 0.1, 0., 0., 0., 0., 0., 0., 0., 0.1,
                          0.1, 0., 0., 0., 0., 0., 0., 0.1, 0.1, 0., 0.,
                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvID2DenseTest, DelimIsSpace) {
  CreateOp(DT_INT64, true, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"  2  10  ", "  ", "8    0  "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {11});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_INT64, {3, 11});
  test::FillValues<int64>(&expected_values, {
                          0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 10,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvID2DenseTest, InvalidDelim) {
  CreateOp(DT_INT64, true, ".");
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, InitOp().code());
}

TEST_F(TransCsvID2DenseTest, InvalidIndex) {
  CreateOp(DT_INT64, true, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2,10", "abc", "8,0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {11});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvID2DenseTest, MissingIndex) {
  CreateOp(DT_INT64, true, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2,10", " ,5", "8,0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {11});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvID2DenseTest, MaxIdNotBigEnough) {
  CreateOp(DT_INT64, true, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {"2,10", "7", "8,0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {8});
  // default value
  AddInputFromArray<int64>(TensorShape({}), {0});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}


class TransCsvKV2SparseTest : public OpsTestBase {
 protected:

  void CreateOp(DataType dtype, string delim) {
    TF_ASSERT_OK(NodeDefBuilder("op", "TransCsvKV2Sparse")
                     .Input(FakeInput(DT_STRING))
                     .Input(FakeInput(DT_INT64))
                     .Attr("T", dtype)
                     .Attr("field_delim", delim)
                     .Finalize(node_def()));
  }
};

TEST_F(TransCsvKV2SparseTest, NormalProcess_int64) {
  CreateOp(DT_INT64, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:1, 2: 2, 4 : 4, 10:10",
                            "3:33, 9:99, 0:22",
                            "2:24, 7:84, 8: 96"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {11});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices(allocator(), DT_INT64, {10, 2});
  test::FillValues<int64>(&expected_indices, {
                          0, 1, 0, 2, 0, 4, 0, 10, 1, 0, 1, 3, 1, 9, 2, 2, 2, 7, 2, 8});
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_INT64, {10});
  test::FillValues<int64>(&expected_values, {1, 2, 4, 10, 22, 33, 99, 24, 84, 96});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {2});
  test::FillValues<int64>(&expected_shape, {3, 11});
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

TEST_F(TransCsvKV2SparseTest, NormalProcess_int32) {
  CreateOp(DT_INT32, "~");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:1~ 2: 2~ 4 : 4",
                            "",
                            "8:99 ~2:24~ 7:84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {10});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices(allocator(), DT_INT64, {6, 2});
  test::FillValues<int64>(&expected_indices, {
                          0, 1, 0, 2, 0, 4, 2, 2, 2, 7, 2, 8});
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_INT32, {6});
  test::FillValues<int32>(&expected_values, {1, 2, 4, 24, 84, 99});
  test::ExpectTensorEqual<int32>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {2});
  test::FillValues<int64>(&expected_shape, {3, 10});
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

TEST_F(TransCsvKV2SparseTest, NormalProcess_float) {
  CreateOp(DT_FLOAT, "~");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({4}), {
                            "1:0.1~ 2: 0.2~ 4 : 0.4~ 10:1.0",
                            "8:+0.99E-01 ~2:-.24~ 7:0.84 ",
                            "",
                            ""});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {-1048575});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices(allocator(), DT_INT64, {7, 2});
  test::FillValues<int64>(&expected_indices, {
                          0, 1, 0, 2, 0, 4, 0, 10, 1, 2, 1, 7, 1, 8});
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_FLOAT, {7});
  test::FillValues<float>(&expected_values, {0.1, 0.2, 0.4, 1.0, -0.24, 0.84, 0.099});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {2});
  test::FillValues<int64>(&expected_shape, {4, 11});
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

TEST_F(TransCsvKV2SparseTest, DelimIsSpace) {
  CreateOp(DT_FLOAT, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({4}), {
                            "1:0.1  2:0.2  4:0.4  10:1.0  ",
                            "   8:+0.99E-01 2:-.24 7:0.84 ",
                            "",
                            "   "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {-1048575});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices(allocator(), DT_INT64, {7, 2});
  test::FillValues<int64>(&expected_indices, {
                          0, 1, 0, 2, 0, 4, 0, 10, 1, 2, 1, 7, 1, 8});
  test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

  Tensor expected_values(allocator(), DT_FLOAT, {7});
  test::FillValues<float>(&expected_values, {0.1, 0.2, 0.4, 1.0, -0.24, 0.84, 0.099});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(1));

  Tensor expected_shape(allocator(), DT_INT64, {2});
  test::FillValues<int64>(&expected_shape, {4, 11});
  test::ExpectTensorEqual<int64>(expected_shape, *GetOutput(2));
}

TEST_F(TransCsvKV2SparseTest, InvalidDelim) {
  CreateOp(DT_FLOAT, "+");
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, InitOp().code());
}

TEST_F(TransCsvKV2SparseTest, InvalidIndex) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1, abc2: 0.2, 4 : 0.4, 10:1.0",
                            "0:0.22, 3:0.33, 9:0.99",
                            "8:0.99 ,2:0.24, 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2SparseTest, InvalidValue) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1, 2: ab.2, 4 : 0.4, 10:1.0",
                            "0:0.22, 3:0.33, 9:0.99",
                            "8:0.99 ,2:0.24, 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2SparseTest, MissingKVSeperator) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1, 2, 4 : 0.4, 10:1.0",
                            "0:0.22, 3:0.33, 9:0.99",
                            "8:0.99 ,2:0.24, 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2SparseTest, KVMissingKey) {
  CreateOp(DT_FLOAT, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1  :2  4:0.4 10:1.0",
                            "0:0.22  3:0.33 9:0.99",
                            "8:0.99  2:0.24  7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2SparseTest, KVMissingValue) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1, 2:, 4 : 0.4, 10:1.0",
                            "0:0.22, 3:0.33, 9:0.99",
                            "8:0.99 ,2:0.24, 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2SparseTest, MaxIdNotBigEnough) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1, 2: 0.2, 4 : 0.4, 10:1.0",
                            "0:0.22, 3:0.33, 9:0.99",
                            "8:0.99 ,2:0.24, 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {8});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}


class TransCsvKV2DenseTest : public OpsTestBase {
 protected:

  void CreateOp(DataType dtype, string delim) {
    TF_ASSERT_OK(NodeDefBuilder("op", "TransCsvKV2Dense")
                     .Input(FakeInput(DT_STRING))
                     .Input(FakeInput(DT_INT64))
                     .Attr("T", dtype)
                     .Attr("field_delim", delim)
                     .Finalize(node_def()));
  }
};

TEST_F(TransCsvKV2DenseTest, NormalProcess_int64) {
  CreateOp(DT_INT64, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:1, 2: 2, 4 : 4, 10:10",
                            "3:33, 9:99, 0:22",
                            "2:24, 7:84, 8: 96"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {-1048575});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_INT64, {3, 11});
  test::FillValues<int64>(&expected_values, {
                             0, 1, 2, 0, 4, 0, 0, 0, 0, 0, 10,
                             22, 0, 0, 33, 0, 0, 0, 0, 0, 99, 0,
                             0, 0, 24, 0, 0, 0, 0, 84, 96, 0, 0});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvKV2DenseTest, NormalProcess_int32) {
  CreateOp(DT_INT32, "~");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:1~ 2: 2~ 4 : 4",
                            "",
                            "8:99 ~2:24~ 7:84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {10});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_INT32, {3, 10});
  test::FillValues<int32>(&expected_values, {
                             0, 1, 2, 0, 4, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 24, 0, 0, 0, 0, 84, 99, 0});
  test::ExpectTensorEqual<int32>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvKV2DenseTest, NormalProcess_float) {
  CreateOp(DT_FLOAT, "~");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({4}), {
                            "1:0.1~ 2: 0.2~ 4 : 0.4~ 10:1.0",
                            "8:+0.99E-01 ~2:-.24~ 7:0.84 ",
                            "",
                            ""});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {11});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_FLOAT, {4, 11});
  test::FillValues<float>(&expected_values, {
                             0., 0.1, 0.2, 0., 0.4, 0., 0., 0., 0., 0., 1.0,
                             0., 0., -0.24, 0., 0., 0., 0., 0.84, 0.099, 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvKV2DenseTest, DelimIsSpace) {
  CreateOp(DT_FLOAT, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({4}), {
                            "  1:0.1 2:0.2  4:0.4   10:1.0 ",
                            "8:+0.99E-01  2:-.24 7:0.84  ",
                            "   ",
                            "   "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {11});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_FLOAT, {4, 11});
  test::FillValues<float>(&expected_values, {
                             0., 0.1, 0.2, 0., 0.4, 0., 0., 0., 0., 0., 1.0,
                             0., 0., -0.24, 0., 0., 0., 0., 0.84, 0.099, 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvKV2DenseTest, InvalidDelim) {
  CreateOp(DT_FLOAT, ":");
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, InitOp().code());
}

TEST_F(TransCsvKV2DenseTest, InvalidIndex) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1, abc2: 0.2, 4 : 0.4, 10:1.0",
                            "0:0.22, 3:0.33, 9:0.99",
                            "8:0.99 ,2:0.24, 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2DenseTest, InvalidValue) {
  CreateOp(DT_FLOAT, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1 2:0.2 4 : 0.4 10:1.0",
                            "0:0.22 3:0.33 9:0.99",
                            "8:0.99 2:0.24 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2DenseTest, MissingKVSeperator) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1, 2, 4 : 0.4, 10:1.0",
                            "0:0.22, 3:0.33, 9:0.99",
                            "8:0.99 ,2:0.24, 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2DenseTest, KVMissingKey) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1, :2, 4 : 0.4, 10:1.0",
                            "0:0.22, 3:0.33, 9:0.99",
                            "8:0.99 ,2:0.24, 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2DenseTest, KVMissingValue) {
  CreateOp(DT_FLOAT, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1 2: 4:0.4 10:1.0",
                            "0:0.22 3:0.33 9:0.99",
                            "8:0.99 2:0.24 7:0.84"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvKV2DenseTest, MaxIdNotBigEnough) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1:0.1, 2: 0.2, 4 : 0.4, 10:1.0",
                            "0:0.22, 3:0.33, 9:0.99",
                            "8:0.99 ,2:0.24, 7:0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {8});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}


class TransCsvToDenseTest : public OpsTestBase {
 protected:

  void CreateOp(DataType dtype, string delim) {
    TF_ASSERT_OK(NodeDefBuilder("op", "TransCsvToDense")
                     .Input(FakeInput(DT_STRING))
                     .Input(FakeInput(DT_INT64))
                     .Attr("T", dtype)
                     .Attr("field_delim", delim)
                     .Finalize(node_def()));
  }
};

TEST_F(TransCsvToDenseTest, NormalProcess_int64) {
  CreateOp(DT_INT64, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1  2   4 10 ",
                            "  ",
                            "  24 84  96"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {7});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_INT64, {3, 7});
  test::FillValues<int64>(&expected_values, {
                             1, 2, 4, 10, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0,
                             24, 84, 96, 0, 0, 0, 0});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvToDenseTest, NormalProcess_int32) {
  CreateOp(DT_INT32, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "1,  -2 ,  4, 10",
                            "",
                            "-24,84, 96, -0"});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {-1048575});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_INT32, {3, 4});
  test::FillValues<int32>(&expected_values, {
                             1, -2, 4, 10,
                             0, 0, 0, 0,
                             -24, 84, 96, 0});
  test::ExpectTensorEqual<int32>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvToDenseTest, NormalProcess_float) {
  CreateOp(DT_FLOAT, "~");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({4}), {
                            "0.1~ 0.2~  0.4~ 1.0",
                            "+0.99E-01 ~-.24~ 0.84 ",
                            "",
                            ""});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {5});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_FLOAT, {4, 5});
  test::FillValues<float>(&expected_values, {
                             0.1, 0.2, 0.4, 1.0, 0.,
                             0.099, -0.24, 0.84, 0., 0.,
                             0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0.});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvToDenseTest, DelimIsSpace) {
  CreateOp(DT_FLOAT, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({4}), {
                            "  0.1 0.2  0.4   1.0 ",
                            "+0.99E-01  -.24 0.84  ",
                            "   ",
                            "   "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {-1048575});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_values(allocator(), DT_FLOAT, {4, 4});
  test::FillValues<float>(&expected_values, {
                             0.1, 0.2, 0.4, 1.0,
                             0.099, -0.24, 0.84, 0.,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.});
  test::ExpectTensorEqual<float>(expected_values, *GetOutput(0));
}

TEST_F(TransCsvToDenseTest, InvalidDelim) {
  CreateOp(DT_FLOAT, "0");
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, InitOp().code());
}

TEST_F(TransCsvToDenseTest, InvalidValue) {
  CreateOp(DT_FLOAT, " ");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "0.1 0.2a  0.4 1.0",
                            "0.22 0.33 0.99",
                            "0.99 0.24 0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvToDenseTest, MissingSeperator) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "0.1, 2, 0.4 1.0",
                            "0.22, 0.33, 0.99",
                            "0.99 ,0.24, 0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {12});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

TEST_F(TransCsvToDenseTest, MaxIdNotBigEnough) {
  CreateOp(DT_FLOAT, ",");
  TF_ASSERT_OK(InitOp());

  // input records
  AddInputFromArray<string>(TensorShape({3}), {
                            "0.1,  0.2,  0.4, 1.0",
                            "0.22, 0.33, 0.99",
                            "0.99 ,0.24, 0.84 "});
  // max_id
  AddInputFromArray<int64>(TensorShape({}), {3});

  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}


template <typename T>
static Graph* CsvID2Sparse(Tensor& records_t) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor zerol(DT_INT64, TensorShape({}));
  zerol.flat<int64>()(0) = static_cast<int64>(-1048575);
  Tensor zerot(DataTypeToEnum<T>::value, TensorShape({}));
  zerot.flat<T>()(0) = static_cast<T>(0);
  Node *node;
  TF_CHECK_OK(NodeBuilder(g->NewName("op"), "TransCsvID2Sparse")
                  .Input(test::graph::Constant(g, records_t))
                  .Input(test::graph::Constant(g, zerol))
                  .Input(test::graph::Constant(g, zerot))
                  .Attr("id_as_value", true)
                  .Attr("field_delim", ",")
                  .Finalize(g, &node));

  return g;
}

template <typename T>
static Graph* CsvID2Dense(Tensor& records_t) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor zerol(DT_INT64, TensorShape({}));
  zerol.flat<int64>()(0) = static_cast<int64>(-1048575);
  Tensor zerot(DataTypeToEnum<T>::value, TensorShape({}));
  zerot.flat<T>()(0) = static_cast<T>(0);
  Node *node;
  TF_CHECK_OK(NodeBuilder(g->NewName("op"), "TransCsvID2Dense")
                  .Input(test::graph::Constant(g, records_t))
                  .Input(test::graph::Constant(g, zerol))
                  .Input(test::graph::Constant(g, zerot))
                  .Attr("id_as_value", false)
                  .Attr("field_delim", ",")
                  .Finalize(g, &node));

  return g;
}

template <typename T>
static Graph* CsvKV2Sparse(Tensor& records_t) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor zerol(DT_INT64, TensorShape({}));
  zerol.flat<int64>()(0) = static_cast<int64>(-1048575);
  Node *node;
  TF_CHECK_OK(NodeBuilder(g->NewName("op"), "TransCsvKV2Sparse")
                  .Input(test::graph::Constant(g, records_t))
                  .Input(test::graph::Constant(g, zerol))
                  .Attr("T", DataTypeToEnum<T>::v())
                  .Attr("field_delim", ",")
                  .Finalize(g, &node));

  return g;
}

template <typename T>
static Graph* CsvKV2Dense(Tensor& records_t) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor zerol(DT_INT64, TensorShape({}));
  zerol.flat<int64>()(0) = static_cast<int64>(-1048575);
  Node *node;
  TF_CHECK_OK(NodeBuilder(g->NewName("op"), "TransCsvKV2Dense")
                  .Input(test::graph::Constant(g, records_t))
                  .Input(test::graph::Constant(g, zerol))
                  .Attr("T", DataTypeToEnum<T>::v())
                  .Attr("field_delim", ",")
                  .Finalize(g, &node));

  return g;
}

}  // namespace

#define BM_INDEX(T, KIND, B, C, S, NTH)                                        \
  static void BM_Csv_ID2##KIND##_##T##_##B##_##C##_##S##_##NTH##t (int iters) {\
    testing::StopTiming();                                                     \
    testing::ItemsProcessed(static_cast<int64>(iters) * B * C);                \
    std::string label(#KIND"_"#T" : Batch "#B);                                \
    testing::SetLabel(label);                                                  \
    testing::UseRealTime();                                                    \
    Tensor records_t(DT_STRING, TensorShape({B}));                             \
    auto records = records_t.flat<string>();                                   \
    for (int i = 0; i < (B); ++i) {                                            \
      std::string line;                                                        \
      for (int j = 0; j < (C) - 1 ; ++j) {                                     \
        line += strings::Printf(" %d ,", j*(S)+1);                             \
      }                                                                        \
      line += strings::Printf(" %d ", ((C)-1)*(S)+1);                          \
      records(i) = line;                                                       \
    }                                                                          \
    auto g = CsvID2##KIND<T>(records_t);                                      \
    SessionOptions opts;                                                       \
    opts.config.set_intra_op_parallelism_threads(NTH);                         \
    opts.config.set_inter_op_parallelism_threads(NTH);                         \
    testing::StartTiming();                                                    \
    test::Benchmark("cpu", g, &opts).Run(iters);                               \
  }                                                                            \
  BENCHMARK(BM_Csv_ID2##KIND##_##T##_##B##_##C##_##S##_##NTH##t);

BM_INDEX(int64, Sparse, 128, 50, 100, 1);
BM_INDEX(int64, Sparse, 1280, 50, 100, 1);
BM_INDEX(int64, Sparse, 2000, 100, 2000, 1);

BM_INDEX(int64, Sparse, 128, 50, 100, 16);
BM_INDEX(int64, Sparse, 1280, 50, 100, 16);
BM_INDEX(int64, Sparse, 2000, 100, 2000, 16);

BM_INDEX(float, Sparse, 128, 50, 100, 1);
BM_INDEX(float, Sparse, 1280, 50, 100, 1);
BM_INDEX(float, Sparse, 2000, 100, 2000, 1);

BM_INDEX(float, Sparse, 128, 50, 100, 16);
BM_INDEX(float, Sparse, 1280, 50, 100, 16);
BM_INDEX(float, Sparse, 2000, 100, 2000, 16);

BM_INDEX(int64, Dense, 128, 50, 4, 1);
BM_INDEX(int64, Dense, 1280, 50, 4, 1);
BM_INDEX(int64, Dense, 2000, 100, 5, 1);

BM_INDEX(int64, Dense, 128, 50, 4, 16);
BM_INDEX(int64, Dense, 1280, 50, 4, 16);
BM_INDEX(int64, Dense, 2000, 100, 5, 16);

BM_INDEX(float, Dense, 128, 50, 4, 1);
BM_INDEX(float, Dense, 1280, 50, 4, 1);
BM_INDEX(float, Dense, 2000, 100, 5, 1);

BM_INDEX(float, Dense, 128, 50, 4, 16);
BM_INDEX(float, Dense, 1280, 50, 4, 16);
BM_INDEX(float, Dense, 2000, 100, 5, 16);

#define BM_KV(T, KIND, B, C, S, NTH)                                           \
  static void BM_Csv_KV2##KIND##_##T##_##B##_##C##_##S##_##NTH##t (int iters) {\
    testing::StopTiming();                                                     \
    testing::ItemsProcessed(static_cast<int64>(iters) * B * C);                \
    std::string label("KV2"#KIND"_"#T" : Batch "#B);                           \
    testing::SetLabel(label);                                                  \
    testing::UseRealTime();                                                    \
    Tensor records_t(DT_STRING, TensorShape({B}));                             \
    auto records = records_t.flat<string>();                                   \
    for (int i = 0; i < (B); ++i) {                                            \
      std::string line;                                                        \
      for (int j = 0; j < (C) - 1 ; ++j) {                                     \
        line += strings::Printf(" %d : -0123456789 ,", j*(S)+1);               \
      }                                                                        \
      line += strings::Printf(" %d : -0123456789", ((C)-1)*(S)+1);             \
      records(i) = line;                                                       \
    }                                                                          \
    auto g = CsvKV2##KIND<T>(records_t);                                       \
    SessionOptions opts;                                                       \
    opts.config.set_intra_op_parallelism_threads(NTH);                         \
    opts.config.set_inter_op_parallelism_threads(NTH);                         \
    testing::StartTiming();                                                    \
    test::Benchmark("cpu", g, &opts).Run(iters);                               \
  }                                                                            \
  BENCHMARK(BM_Csv_KV2##KIND##_##T##_##B##_##C##_##S##_##NTH##t);

BM_KV(float, Sparse, 128, 338, 4, 1);
BM_KV(float, Sparse, 1280, 338, 4, 1);
BM_KV(float, Sparse, 2000, 338, 5, 1);

BM_KV(float, Sparse, 128, 338, 4, 16);
BM_KV(float, Sparse, 1280, 338, 4, 16);
BM_KV(float, Sparse, 2000, 338, 5, 16);

BM_KV(float, Dense, 128, 338, 4, 1);
BM_KV(float, Dense, 1280, 338, 4, 1);
BM_KV(float, Dense, 2000, 338, 5, 1);

BM_KV(float, Dense, 128, 338, 4, 16);
BM_KV(float, Dense, 1280, 338, 4, 16);
BM_KV(float, Dense, 2000, 338, 5, 16);

}  // namespace tensorflow

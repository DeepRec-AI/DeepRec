/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/util/tensor_bundle/db_writer.h"

#include <random>
#include <vector>

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap.h"

namespace tensorflow {

namespace {

// Prepend the current test case's working temporary directory to <prefix>
string Prefix(const string& prefix) {
  return strings::StrCat(testing::TmpDir(), "/", prefix);
}

// Construct a data input directory by prepending the test data root
// directory to <prefix>
string TestdataPrefix(const string& prefix) {
  return strings::StrCat(testing::TensorFlowSrcRoot(),
                         "/core/util/tensor_bundle/testdata/", prefix);
}

template <typename T>
Tensor Constant(T v, TensorShape shape) {
  Tensor ret(DataTypeToEnum<T>::value, shape);
  ret.flat<T>().setConstant(v);
  return ret;
}

template <typename T>
Tensor Constant_2x3(T v) {
  return Constant(v, TensorShape({2, 3}));
}

Tensor ByteSwap(Tensor t) {
  Tensor ret = tensor::DeepCopy(t);
  TF_EXPECT_OK(ByteSwapTensor(&ret));
  return ret;
}

// Assert that <reader> has a tensor under <key> matching <expected_val> in
// terms of both shape, dtype, and value
template <typename T>
void Expect(DBReader* reader, const string& key,
            const Tensor& expected_val) {
  // Tests for Contains().
  EXPECT_TRUE(reader->Contains(key));
  // Tests for LookupDtypeAndShape().
  DataType dtype;
  TensorShape shape;
  TF_ASSERT_OK(reader->LookupDtypeAndShape(key, &dtype, &shape));
  EXPECT_EQ(expected_val.dtype(), dtype);
  EXPECT_EQ(expected_val.shape(), shape);
  // Tests for Lookup(), checking tensor contents.
  Tensor val(expected_val.dtype(), shape);
  TF_ASSERT_OK(reader->Lookup(key, &val));
  test::ExpectTensorEqual<T>(val, expected_val);
}

template <class T>
void ExpectVariant(DBReader* reader, const string& key,
                   const Tensor& expected_t) {
  // Tests for Contains().
  EXPECT_TRUE(reader->Contains(key));
  // Tests for LookupDtypeAndShape().
  DataType dtype;
  TensorShape shape;
  TF_ASSERT_OK(reader->LookupDtypeAndShape(key, &dtype, &shape));
  // Tests for Lookup(), checking tensor contents.
  EXPECT_EQ(expected_t.dtype(), dtype);
  EXPECT_EQ(expected_t.shape(), shape);
  Tensor actual_t(dtype, shape);
  TF_ASSERT_OK(reader->Lookup(key, &actual_t));
  for (int i = 0; i < expected_t.NumElements(); i++) {
    Variant actual_var = actual_t.flat<Variant>()(i);
    Variant expected_var = expected_t.flat<Variant>()(i);
    EXPECT_EQ(actual_var.TypeName(), expected_var.TypeName());
    auto* actual_val = actual_var.get<T>();
    auto* expected_val = expected_var.get<T>();
    EXPECT_EQ(*expected_val, *actual_val);
  }
}

template <typename T>
void ExpectNext(DBReader* reader, const Tensor& expected_val) {
  EXPECT_TRUE(reader->Valid());
  reader->Next();
  TF_ASSERT_OK(reader->status());
  Tensor val;
  TF_ASSERT_OK(reader->ReadCurrent(&val));
  test::ExpectTensorEqual<T>(val, expected_val);
}

std::vector<string> AllTensorKeys(DBReader* reader) {
  std::vector<string> ret;
  reader->Seek(kHeaderEntryKey);
  reader->Next();
  for (; reader->Valid(); reader->Next()) {
    ret.emplace_back(reader->key());
  }
  return ret;
}

// Writes out the metadata file of a bundle again, with the endianness marker
// bit flipped.
Status FlipEndiannessBit(const string& prefix) {
  Env* env = Env::Default();
  const string metadata_tmp_path = Prefix("some_tmp_path");
  std::unique_ptr<WritableFile> metadata_file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(metadata_tmp_path, &metadata_file));
  // We create the builder lazily in case we run into an exception earlier, in
  // which case we'd forget to call Finish() and TableBuilder's destructor
  // would complain.
  std::unique_ptr<table::TableBuilder> builder;

  // Reads the existing metadata file, and fills the builder.
  {
    const string filename = MetaFilename(prefix);
    uint64 file_size;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    table::Table* table = nullptr;
    TF_RETURN_IF_ERROR(
        table::Table::Open(table::Options(), file.get(), file_size, &table));
    std::unique_ptr<table::Table> table_deleter(table);
    std::unique_ptr<table::Iterator> iter(table->NewIterator());

    // Reads the header entry.
    iter->Seek(kHeaderEntryKey);
    CHECK(iter->Valid());
    BundleHeaderProto header;
    CHECK(header.ParseFromArray(iter->value().data(), iter->value().size()));
    // Flips the endianness.
    if (header.endianness() == BundleHeaderProto::LITTLE) {
      header.set_endianness(BundleHeaderProto::BIG);
    } else {
      header.set_endianness(BundleHeaderProto::LITTLE);
    }
    builder.reset(
        new table::TableBuilder(table::Options(), metadata_file.get()));
    builder->Add(iter->key(), header.SerializeAsString());
    iter->Next();

    // Adds the non-header entries unmodified.
    for (; iter->Valid(); iter->Next())
      builder->Add(iter->key(), iter->value());
  }
  TF_RETURN_IF_ERROR(builder->Finish());
  TF_RETURN_IF_ERROR(env->RenameFile(metadata_tmp_path, MetaFilename(prefix)));
  return metadata_file->Close();
}

template <typename T>
void TestBasic() {
  {
    DBWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(writer.Add("foo_003", Constant_2x3(T(3))));
    TF_EXPECT_OK(writer.Add("foo_000", Constant_2x3(T(0))));
    TF_EXPECT_OK(writer.Add("foo_002", Constant_2x3(T(2))));
    TF_EXPECT_OK(writer.Add("foo_001", Constant_2x3(T(1))));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    DBReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "foo_000", Constant_2x3(T(0)));
    Expect<T>(&reader, "foo_001", Constant_2x3(T(1)));
    Expect<T>(&reader, "foo_002", Constant_2x3(T(2)));
    Expect<T>(&reader, "foo_003", Constant_2x3(T(3)));
  }
  {
    DBReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3(T(0)));
    ExpectNext<T>(&reader, Constant_2x3(T(1)));
    ExpectNext<T>(&reader, Constant_2x3(T(2)));
    ExpectNext<T>(&reader, Constant_2x3(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  {
    DBWriter writer(Env::Default(), Prefix("bar"));
    TF_EXPECT_OK(writer.Add("bar_003", Constant_2x3(T(3))));
    TF_EXPECT_OK(writer.Add("bar_000", Constant_2x3(T(0))));
    TF_EXPECT_OK(writer.Add("bar_002", Constant_2x3(T(2))));
    TF_EXPECT_OK(writer.Add("bar_001", Constant_2x3(T(1))));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    DBReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003"}));
    Expect<T>(&reader, "bar_003", Constant_2x3(T(3)));
    Expect<T>(&reader, "bar_002", Constant_2x3(T(2)));
    Expect<T>(&reader, "bar_001", Constant_2x3(T(1)));
    Expect<T>(&reader, "bar_000", Constant_2x3(T(0)));
  }
  {
    DBReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3(T(0)));
    ExpectNext<T>(&reader, Constant_2x3(T(1)));
    ExpectNext<T>(&reader, Constant_2x3(T(2)));
    ExpectNext<T>(&reader, Constant_2x3(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  TF_ASSERT_OK(MergeBundles(Env::Default(), {Prefix("foo"), Prefix("bar")},
                            Prefix("merged")));
  {
    DBReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003",
                             "foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "bar_000", Constant_2x3(T(0)));
    Expect<T>(&reader, "bar_001", Constant_2x3(T(1)));
    Expect<T>(&reader, "bar_002", Constant_2x3(T(2)));
    Expect<T>(&reader, "bar_003", Constant_2x3(T(3)));
    Expect<T>(&reader, "foo_000", Constant_2x3(T(0)));
    Expect<T>(&reader, "foo_001", Constant_2x3(T(1)));
    Expect<T>(&reader, "foo_002", Constant_2x3(T(2)));
    Expect<T>(&reader, "foo_003", Constant_2x3(T(3)));
  }
  {
    DBReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3(T(0)));
    ExpectNext<T>(&reader, Constant_2x3(T(1)));
    ExpectNext<T>(&reader, Constant_2x3(T(2)));
    ExpectNext<T>(&reader, Constant_2x3(T(3)));
    ExpectNext<T>(&reader, Constant_2x3(T(0)));
    ExpectNext<T>(&reader, Constant_2x3(T(1)));
    ExpectNext<T>(&reader, Constant_2x3(T(2)));
    ExpectNext<T>(&reader, Constant_2x3(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
}

// Type-specific subroutine of SwapBytes test below
template <typename T>
void TestByteSwap(const T* forward, const T* swapped, int array_len) {
  auto bytes_per_elem = sizeof(T);

  // Convert the entire array at once
  std::unique_ptr<T[]> forward_copy(new T[array_len]);
  std::memcpy(forward_copy.get(), forward, array_len * bytes_per_elem);
  TF_EXPECT_OK(ByteSwapArray(reinterpret_cast<char*>(forward_copy.get()),
                             bytes_per_elem, array_len));
  for (int i = 0; i < array_len; i++) {
    EXPECT_EQ(forward_copy.get()[i], swapped[i]);
  }

  // Then the array wrapped in a tensor
  auto shape = TensorShape({array_len});
  auto dtype = DataTypeToEnum<T>::value;
  Tensor forward_tensor(dtype, shape);
  Tensor swapped_tensor(dtype, shape);
  std::memcpy(const_cast<char*>(forward_tensor.tensor_data().data()), forward,
              array_len * bytes_per_elem);
  std::memcpy(const_cast<char*>(swapped_tensor.tensor_data().data()), swapped,
              array_len * bytes_per_elem);
  TF_EXPECT_OK(ByteSwapTensor(&forward_tensor));
  test::ExpectTensorEqual<T>(forward_tensor, swapped_tensor);
}

// Unit test of the byte-swapping operations that TensorBundle uses.
TEST(TensorBundleTest, SwapBytes) {
  // A bug in the compiler on MacOS causes ByteSwap() and FlipEndiannessBit()
  // to be removed from the executable if they are only called from templated
  // functions. As a workaround, we make some dummy calls here.
  // TODO(frreiss): Remove this workaround when the compiler bug is fixed.
  ByteSwap(Constant_2x3<int>(42));
  EXPECT_NE(Status::OK(), FlipEndiannessBit(Prefix("not_a_valid_prefix")));

  // Test patterns, manually swapped so that we aren't relying on the
  // correctness of our own byte-swapping macros when testing those macros.
  // At least one of the entries in each list has the sign bit set when
  // interpreted as a signed int.
  const int arr_len_16 = 4;
  const uint16_t forward_16[] = {0x1de5, 0xd017, 0xf1ea, 0xc0a1};
  const uint16_t swapped_16[] = {0xe51d, 0x17d0, 0xeaf1, 0xa1c0};
  const int arr_len_32 = 2;
  const uint32_t forward_32[] = {0x0ddba115, 0xf01dab1e};
  const uint32_t swapped_32[] = {0x15a1db0d, 0x1eab1df0};
  const int arr_len_64 = 2;
  const uint64_t forward_64[] = {0xf005ba11caba1000, 0x5ca1ab1ecab005e5};
  const uint64_t swapped_64[] = {0x0010baca11ba05f0, 0xe505b0ca1eaba15c};

  // 16-bit types
  TestByteSwap(forward_16, swapped_16, arr_len_16);
  TestByteSwap(reinterpret_cast<const int16_t*>(forward_16),
               reinterpret_cast<const int16_t*>(swapped_16), arr_len_16);
  TestByteSwap(reinterpret_cast<const bfloat16*>(forward_16),
               reinterpret_cast<const bfloat16*>(swapped_16), arr_len_16);

  // 32-bit types
  TestByteSwap(forward_32, swapped_32, arr_len_32);
  TestByteSwap(reinterpret_cast<const int32_t*>(forward_32),
               reinterpret_cast<const int32_t*>(swapped_32), arr_len_32);
  TestByteSwap(reinterpret_cast<const float*>(forward_32),
               reinterpret_cast<const float*>(swapped_32), arr_len_32);

  // 64-bit types
  // Cast to uint64*/int64* to make DataTypeToEnum<T> happy
  TestByteSwap(reinterpret_cast<const uint64*>(forward_64),
               reinterpret_cast<const uint64*>(swapped_64), arr_len_64);
  TestByteSwap(reinterpret_cast<const int64*>(forward_64),
               reinterpret_cast<const int64*>(swapped_64), arr_len_64);
  TestByteSwap(reinterpret_cast<const double*>(forward_64),
               reinterpret_cast<const double*>(swapped_64), arr_len_64);

  // Complex types.
  // Logic for complex number handling is only in ByteSwapTensor, so don't test
  // ByteSwapArray
  const float* forward_float = reinterpret_cast<const float*>(forward_32);
  const float* swapped_float = reinterpret_cast<const float*>(swapped_32);
  const double* forward_double = reinterpret_cast<const double*>(forward_64);
  const double* swapped_double = reinterpret_cast<const double*>(swapped_64);
  Tensor forward_complex64 = Constant_2x3<complex64>(
      std::complex<float>(forward_float[0], forward_float[1]));
  Tensor swapped_complex64 = Constant_2x3<complex64>(
      std::complex<float>(swapped_float[0], swapped_float[1]));
  Tensor forward_complex128 = Constant_2x3<complex128>(
      std::complex<double>(forward_double[0], forward_double[1]));
  Tensor swapped_complex128 = Constant_2x3<complex128>(
      std::complex<double>(swapped_double[0], swapped_double[1]));

  TF_EXPECT_OK(ByteSwapTensor(&forward_complex64));
  test::ExpectTensorEqual<complex64>(forward_complex64, swapped_complex64);

  TF_EXPECT_OK(ByteSwapTensor(&forward_complex128));
  test::ExpectTensorEqual<complex128>(forward_complex128, swapped_complex128);
}

// Basic test of alternate-endianness support. Generates a bundle in
// the opposite of the current system's endianness and attempts to
// read the bundle back in. Does not exercise sharding or access to
// nonaligned tensors. Does cover the major access types exercised
// in TestBasic.
template <typename T>
void TestEndianness() {
  {
    // Write out a TensorBundle in the opposite of this host's endianness.
    DBWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(writer.Add("foo_003", ByteSwap(Constant_2x3<T>(T(3)))));
    TF_EXPECT_OK(writer.Add("foo_000", ByteSwap(Constant_2x3<T>(T(0)))));
    TF_EXPECT_OK(writer.Add("foo_002", ByteSwap(Constant_2x3<T>(T(2)))));
    TF_EXPECT_OK(writer.Add("foo_001", ByteSwap(Constant_2x3<T>(T(1)))));
    TF_ASSERT_OK(writer.Finish());
    TF_ASSERT_OK(FlipEndiannessBit(Prefix("foo")));
  }
  {
    DBReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "foo_000", Constant_2x3<T>(T(0)));
    Expect<T>(&reader, "foo_001", Constant_2x3<T>(T(1)));
    Expect<T>(&reader, "foo_002", Constant_2x3<T>(T(2)));
    Expect<T>(&reader, "foo_003", Constant_2x3<T>(T(3)));
  }
  {
    DBReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3<T>(T(0)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(1)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(2)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  {
    DBWriter writer(Env::Default(), Prefix("bar"));
    TF_EXPECT_OK(writer.Add("bar_003", ByteSwap(Constant_2x3<T>(T(3)))));
    TF_EXPECT_OK(writer.Add("bar_000", ByteSwap(Constant_2x3<T>(T(0)))));
    TF_EXPECT_OK(writer.Add("bar_002", ByteSwap(Constant_2x3<T>(T(2)))));
    TF_EXPECT_OK(writer.Add("bar_001", ByteSwap(Constant_2x3<T>(T(1)))));
    TF_ASSERT_OK(writer.Finish());
    TF_ASSERT_OK(FlipEndiannessBit(Prefix("bar")));
  }
  {
    DBReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003"}));
    Expect<T>(&reader, "bar_003", Constant_2x3<T>(T(3)));
    Expect<T>(&reader, "bar_002", Constant_2x3<T>(T(2)));
    Expect<T>(&reader, "bar_001", Constant_2x3<T>(T(1)));
    Expect<T>(&reader, "bar_000", Constant_2x3<T>(T(0)));
  }
  {
    DBReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3<T>(T(0)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(1)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(2)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  TF_ASSERT_OK(MergeBundles(Env::Default(), {Prefix("foo"), Prefix("bar")},
                            Prefix("merged")));
  {
    DBReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003",
                             "foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "bar_000", Constant_2x3<T>(T(0)));
    Expect<T>(&reader, "bar_001", Constant_2x3<T>(T(1)));
    Expect<T>(&reader, "bar_002", Constant_2x3<T>(T(2)));
    Expect<T>(&reader, "bar_003", Constant_2x3<T>(T(3)));
    Expect<T>(&reader, "foo_000", Constant_2x3<T>(T(0)));
    Expect<T>(&reader, "foo_001", Constant_2x3<T>(T(1)));
    Expect<T>(&reader, "foo_002", Constant_2x3<T>(T(2)));
    Expect<T>(&reader, "foo_003", Constant_2x3<T>(T(3)));
  }
  {
    DBReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3<T>(T(0)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(1)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(2)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(3)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(0)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(1)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(2)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
}

template <typename T>
void TestNonStandardShapes() {
  {
    DBWriter writer(Env::Default(), Prefix("nonstandard"));
    TF_EXPECT_OK(writer.Add("scalar", Constant(T(0), TensorShape())));
    TF_EXPECT_OK(
        writer.Add("non_standard0", Constant(T(0), TensorShape({0, 1618}))));
    TF_EXPECT_OK(
        writer.Add("non_standard1", Constant(T(0), TensorShape({16, 0, 18}))));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    DBReader reader(Env::Default(), Prefix("nonstandard"));
    TF_ASSERT_OK(reader.status());
    Expect<T>(&reader, "scalar", Constant(T(0), TensorShape()));
    Expect<T>(&reader, "non_standard0", Constant(T(0), TensorShape({0, 1618})));
    Expect<T>(&reader, "non_standard1",
              Constant(T(0), TensorShape({16, 0, 18})));
  }
}

// Writes a bundle to disk with a bad "version"; checks for "expected_error".
void VersionTest(const VersionDef& version, StringPiece expected_error) {
  const string path = Prefix("version_test");
  {
    // Prepare an empty bundle with the given version information.
    BundleHeaderProto header;
    *header.mutable_version() = version;

    // Write the metadata file to disk.
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(Env::Default()->NewWritableFile(MetaFilename(path), &file));
    table::TableBuilder builder(table::Options(), file.get());
    builder.Add(kHeaderEntryKey, header.SerializeAsString());
    TF_ASSERT_OK(builder.Finish());
  }
  // Read it back in and verify that we get the expected error.
  DBReader reader(Env::Default(), path);
  EXPECT_TRUE(errors::IsInvalidArgument(reader.status()));
  EXPECT_TRUE(
      absl::StartsWith(reader.status().error_message(), expected_error));
}

}  // namespace



TEST(TensorBundleTest, StringTensors) {
  constexpr size_t kLongLength = static_cast<size_t>(UINT32_MAX) + 1;
  Tensor long_string_tensor(DT_STRING, TensorShape({1}));

  {
    DBWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(writer.Add("string_tensor",
                            Tensor(DT_STRING, TensorShape({1}))));  // Empty.
    TF_EXPECT_OK(writer.Add("scalar", test::AsTensor<tstring>({"hello"})));
    TF_EXPECT_OK(writer.Add(
        "strs",
        test::AsTensor<tstring>({"hello", "", "x01", string(1 << 25, 'c')})));

    // Requires a 64-bit length.
    tstring* backing_string = long_string_tensor.flat<tstring>().data();
#ifdef USE_TSTRING
    backing_string->resize_uninitialized(kLongLength);
    std::char_traits<char>::assign(backing_string->data(), kLongLength, 'd');
#else   // USE_TSTRING
    backing_string->assign(kLongLength, 'd');
#endif  // USE_TSTRING
    TF_EXPECT_OK(writer.Add("long_scalar", long_string_tensor));

    // Mixes in some floats.
    TF_EXPECT_OK(writer.Add("floats", Constant_2x3<float>(16.18)));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    DBReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(AllTensorKeys(&reader),
              std::vector<string>({"floats", "long_scalar", "scalar",
                                   "string_tensor", "strs"}));

    Expect<tstring>(&reader, "string_tensor",
                    Tensor(DT_STRING, TensorShape({1})));
    Expect<tstring>(&reader, "scalar", test::AsTensor<tstring>({"hello"}));
    Expect<tstring>(
        &reader, "strs",
        test::AsTensor<tstring>({"hello", "", "x01", string(1 << 25, 'c')}));

    Expect<float>(&reader, "floats", Constant_2x3<float>(16.18));

    // We don't use the Expect function so we can re-use the
    // `long_string_tensor` buffer for reading out long_scalar to keep memory
    // usage reasonable.
    EXPECT_TRUE(reader.Contains("long_scalar"));
    DataType dtype;
    TensorShape shape;
    TF_ASSERT_OK(reader.LookupDtypeAndShape("long_scalar", &dtype, &shape));
    EXPECT_EQ(DT_STRING, dtype);
    EXPECT_EQ(TensorShape({1}), shape);

    // Zero-out the string so that we can be sure the new one is read in.
    tstring* backing_string = long_string_tensor.flat<tstring>().data();
    backing_string->assign("");

    // Read long_scalar and check it contains kLongLength 'd's.
    TF_ASSERT_OK(reader.Lookup("long_scalar", &long_string_tensor));
    ASSERT_EQ(backing_string, long_string_tensor.flat<tstring>().data());
    EXPECT_EQ(kLongLength, backing_string->length());
    for (size_t i = 0; i < kLongLength; i++) {
      // Not using ASSERT_EQ('d', c) because this way is twice as fast due to
      // compiler optimizations.
      if ((*backing_string)[i] != 'd') {
        FAIL() << "long_scalar is not full of 'd's as expected.";
        break;
      }
    }
  }
}

class VariantObject {
 public:
  VariantObject() {}
  VariantObject(const string& metadata, int64 value)
      : metadata_(metadata), value_(value) {}

  string TypeName() const { return "TEST VariantObject"; }
  void Encode(VariantTensorData* data) const {
    data->set_type_name(TypeName());
    data->set_metadata(metadata_);
    Tensor val_t = Tensor(DT_INT64, TensorShape({}));
    val_t.scalar<int64>()() = value_;
    *(data->add_tensors()) = val_t;
  }
  bool Decode(const VariantTensorData& data) {
    EXPECT_EQ(data.type_name(), TypeName());
    data.get_metadata(&metadata_);
    EXPECT_EQ(data.tensors_size(), 1);
    value_ = data.tensors(0).scalar<int64>()();
    return true;
  }
  bool operator==(const VariantObject other) const {
    return metadata_ == other.metadata_ && value_ == other.value_;
  }
  string metadata_;
  int64 value_;
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(VariantObject, "TEST VariantObject");

TEST(TensorBundleTest, VariantTensors) {
  {
    DBWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(
        writer.Add("variant_tensor",
                   test::AsTensor<Variant>({VariantObject("test", 10),
                                            VariantObject("test1", 20)})));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    DBReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectVariant<VariantObject>(
        &reader, "variant_tensor",
        test::AsTensor<Variant>(
            {VariantObject("test", 10), VariantObject("test1", 20)}));
  }
}


class TensorBundleAlignmentTest : public ::testing::Test {
 protected:
  template <typename T>
  void ExpectAlignment(DBReader* reader, const string& key, int alignment) {
    BundleEntryProto full_tensor_entry;
    TF_ASSERT_OK(reader->GetBundleEntryProto(key, &full_tensor_entry));
    EXPECT_EQ(0, full_tensor_entry.offset() % alignment);
  }
};



static void BM_BundleAlignmentByteOff(int iters, int alignment,
                                      int tensor_size) {
  testing::StopTiming();
  {
    DBWriter::Options opts;
    opts.data_alignment = alignment;
    DBWriter writer(Env::Default(), Prefix("foo"), opts);
    TF_CHECK_OK(writer.Add("small", Constant(true, TensorShape({1}))));
    TF_CHECK_OK(writer.Add("big", Constant(32.1, TensorShape({tensor_size}))));
    TF_CHECK_OK(writer.Finish());
  }
  DBReader reader(Env::Default(), Prefix("foo"));
  TF_CHECK_OK(reader.status());
  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    Tensor t;
    TF_CHECK_OK(reader.Lookup("big", &t));
  }
  testing::StopTiming();
}

#define BM_BundleAlignment(ALIGN, SIZE)                        \
  static void BM_BundleAlignment_##ALIGN##_##SIZE(int iters) { \
    BM_BundleAlignmentByteOff(iters, ALIGN, SIZE);             \
  }                                                            \
  BENCHMARK(BM_BundleAlignment_##ALIGN##_##SIZE)

BM_BundleAlignment(1, 512);
BM_BundleAlignment(1, 4096);
BM_BundleAlignment(1, 1048576);
BM_BundleAlignment(4096, 512);
BM_BundleAlignment(4096, 4096);
BM_BundleAlignment(4096, 1048576);

}  // namespace tensorflow

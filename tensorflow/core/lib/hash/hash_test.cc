/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <farmhash.h>
#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

TEST(Hash, SignedUnsignedIssue) {
  const unsigned char d1[1] = {0x62};
  const unsigned char d2[2] = {0xc3, 0x97};
  const unsigned char d3[3] = {0xe2, 0x99, 0xa5};
  const unsigned char d4[4] = {0xe1, 0x80, 0xb9, 0x32};
  const unsigned char d5[48] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  struct Case {
    uint32 hash32;
    uint64 hash64;
    const unsigned char* data;
    size_t size;
    uint32 seed;
  };

  for (Case c : std::vector<Case>{
           {0x471a8188u, 0x4c61ea3eeda4cb87ull, nullptr, 0, 0xbc9f1d34},
           {0xd615eba5u, 0x091309f7ef916c8aull, d1, sizeof(d1), 0xbc9f1d34},
           {0x0c3cccdau, 0xa815bcdf1d1af01cull, d2, sizeof(d2), 0xbc9f1d34},
           {0x3ba37e0eu, 0x02167564e4d06430ull, d3, sizeof(d3), 0xbc9f1d34},
           {0x16174eb3u, 0x8f7ed82ffc21071full, d4, sizeof(d4), 0xbc9f1d34},
           {0x98b1926cu, 0xce196580c97aff1eull, d5, sizeof(d5), 0x12345678},
       }) {
    EXPECT_EQ(c.hash32,
              Hash32(reinterpret_cast<const char*>(c.data), c.size, c.seed));
    EXPECT_EQ(c.hash64,
              Hash64(reinterpret_cast<const char*>(c.data), c.size, c.seed));

    // Check hashes with inputs aligned differently.
    for (int align = 1; align <= 7; align++) {
      std::string input(align, 'x');
      input.append(reinterpret_cast<const char*>(c.data), c.size);
      EXPECT_EQ(c.hash32, Hash32(&input[align], c.size, c.seed));
      EXPECT_EQ(c.hash64, Hash64(&input[align], c.size, c.seed));
    }
  }
}

TEST(Hash, SignedUnsignedIssueHash64V3) {
  const unsigned char d1[1] = {0x62};
  const unsigned char d2[2] = {0xc3, 0x97};
  const unsigned char d3[3] = {0xe2, 0x99, 0xa5};
  const unsigned char d4[4] = {0xe1, 0x80, 0xb9, 0x32};
  const unsigned char d5[48] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  const unsigned char d6[72] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  struct Case {
    uint64 hash64;
    const unsigned char* data;
    size_t size;
    uint32 seed;
  };

  for (Case c : std::vector<Case>{
           {0x4c61ea3eeda4cb87ull, nullptr, 0, 0xbc9f1d34},
           {0x091309f7ef916c8aull, d1, sizeof(d1), 0xbc9f1d34},
           {0xa815bcdf1d1af01cull, d2, sizeof(d2), 0xbc9f1d34},
           {0x02167564e4d06430ull, d3, sizeof(d3), 0xbc9f1d34},
           {0x8f7ed82ffc21071full, d4, sizeof(d4), 0xbc9f1d34},
           {0xce196580c97aff1eull, d5, sizeof(d5), 0x12345678},
           {0x44c8b83651a6c80c, d6, sizeof(d6), 0x12345678},
       }) {
    
    EXPECT_EQ(c.hash64,
              Hash64V3(reinterpret_cast<const char*>(c.data), c.size, c.seed));

    // Check hashes with inputs aligned differently.
    for (int align = 1; align <= 7; align++) {
      std::string input(align, 'x');
      input.append(reinterpret_cast<const char*>(c.data), c.size);
      EXPECT_EQ(c.hash64, Hash64V3(&input[align], c.size, c.seed));
    }
  }
}
TEST(Hash, SignedUnsignedIssueHash64Farm) {
  // const unsigned char d1[1] = {0x62};
  const unsigned char d1[1] = {'a'};
  const unsigned char d2[2] = {0xc3, 0x97};
  const unsigned char d3[3] = {0xe2, 0x99, 0xa5};
  const unsigned char d4[4] = {0xe1, 0x80, 0xb9, 0x32};
  const unsigned char d5[9] = {0xe1, 0x80, 0xb9, 0x32, 0xe1, 0x80, 
                               0xb9, 0x32, 0x62};
  const unsigned char d6[32] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18,
  };
  const unsigned char d7[36] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
  };
  const unsigned char d8[72] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
  };
  struct Case {
    uint64 hash64;
    const unsigned char* data;
    size_t size;
    uint32 seed;
  };
  for (Case c : std::vector<Case>{
           {12917804110809363939, d1, sizeof(d1), 0xbc9f1d34},
           {7333056145295358442, d3, sizeof(d3), 0xbc9f1d34},
           {6652747580939975467, d4, sizeof(d4), 0xbc9f1d34},
           {9354346843959778753, d5, sizeof(d5), 0x12345678},
           {5717533644398455540, d6, sizeof(d6), 0xbc9f1d34},
           {2514457852006695136, d7, sizeof(d7), 0xbc9f1d34},
           {12944174544826175378, d8, sizeof(d8), 0xbc9f1d34},
       }) {
    
    EXPECT_EQ(c.hash64,
              ::util::Fingerprint64(reinterpret_cast<const char*>(c.data), 
                                    c.size));
  }
}

#if defined(__AVX512F__)
TEST(Hash, SignedUnsignedIssueHash64V3Batch) {
  const unsigned char d1[1] = {0x62};
  const unsigned char d2[2] = {0xc3, 0x97};
  const unsigned char d3[3] = {0xe2, 0x99, 0xa5};
  const unsigned char d4[4] = {0xe1, 0x80, 0xb9, 0x32};
  const unsigned char d5[48] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  struct Case {
    // uint32 hash32;
    uint64 hash64;
    const unsigned char* data;
    size_t size;
    uint32 seed;
  };

  for (Case c : std::vector<Case>{
           {0x4c61ea3eeda4cb87ull, nullptr, 0, 0xbc9f1d34},
           {0x091309f7ef916c8aull, d1, sizeof(d1), 0xbc9f1d34},
           {0xa815bcdf1d1af01cull, d2, sizeof(d2), 0xbc9f1d34},
           {0x02167564e4d06430ull, d3, sizeof(d3), 0xbc9f1d34},
           {0x8f7ed82ffc21071full, d4, sizeof(d4), 0xbc9f1d34},
           {0xce196580c97aff1eull, d5, sizeof(d5), 0x12345678}
       }) {
    
    // const char** data_batch = new const char*[4]; 
    const char* data_batch[8];
    data_batch[0] = reinterpret_cast<const char*>(c.data);
    data_batch[1] = reinterpret_cast<const char*>(c.data);
    data_batch[2] = reinterpret_cast<const char*>(c.data);
    data_batch[3] = reinterpret_cast<const char*>(c.data);
    data_batch[4] = reinterpret_cast<const char*>(c.data);
    data_batch[5] = reinterpret_cast<const char*>(c.data);
    data_batch[6] = reinterpret_cast<const char*>(c.data);
    data_batch[7] = reinterpret_cast<const char*>(c.data);
    uint64 h_value[8];
    Hash64V3_Batch512(data_batch, &h_value[0], c.size, c.seed);
    EXPECT_EQ(c.hash64, h_value[0]);
    EXPECT_EQ(c.hash64, h_value[1]);
    EXPECT_EQ(c.hash64, h_value[2]);
    EXPECT_EQ(c.hash64, h_value[3]);
    EXPECT_EQ(c.hash64, h_value[4]);
    EXPECT_EQ(c.hash64, h_value[5]);
    EXPECT_EQ(c.hash64, h_value[6]);
    EXPECT_EQ(c.hash64, h_value[7]);
    // Check hashes with inputs aligned differently.
    for (int align = 1; align <= 7; align++) {
      std::string input(align, 'x');
      input.append(reinterpret_cast<const char*>(c.data), c.size);
      data_batch[0] = (&input[align]);
      data_batch[1] = (&input[align]);
      data_batch[2] = (&input[align]);
      data_batch[3] = (&input[align]);
      data_batch[4] = (&input[align]);
      data_batch[5] = (&input[align]);
      data_batch[6] = (&input[align]);
      data_batch[7] = (&input[align]);
      Hash64V3_Batch512(data_batch, &h_value[0], c.size, c.seed);
      EXPECT_EQ(c.hash64, h_value[0]);
      EXPECT_EQ(c.hash64, h_value[1]);
      EXPECT_EQ(c.hash64, h_value[2]);
      EXPECT_EQ(c.hash64, h_value[3]);
      EXPECT_EQ(c.hash64, h_value[4]);
      EXPECT_EQ(c.hash64, h_value[5]);
      EXPECT_EQ(c.hash64, h_value[6]);
      EXPECT_EQ(c.hash64, h_value[7]);
    }
  }
}

TEST(Hash, SignedUnsignedIssueHash64FarmBatch) {
  // const unsigned char d1[1] = {0x62};
  const unsigned char d1[1] = {'a'};
  const unsigned char d3[3] = {0xe2, 0x99, 0xa5};
  const unsigned char d4[4] = {0xe1, 0x80, 0xb9, 0x32};
  const unsigned char d5[9] = {0xe1, 0x80, 0xb9, 0x32, 0xe1, 
                               0x80, 0xb9, 0x32, 0x62};
  const unsigned char d6[32] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18,
  };
  const unsigned char d7[36] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
  };
  const unsigned char d8[72] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
  };

  struct Case {
    uint64_t hash64;
    const unsigned char* data;
    size_t size;
    uint32 seed;
  };
  for (Case c : std::vector<Case>{
           {12917804110809363939, d1, sizeof(d1), 0xbc9f1d34},
           {7333056145295358442, d3, sizeof(d3), 0xbc9f1d34},
           {6652747580939975467, d4, sizeof(d4), 0xbc9f1d34},
           {9354346843959778753, d5, sizeof(d5), 0x12345678},
           {5717533644398455540, d6, sizeof(d6), 0xbc9f1d34},
           {2514457852006695136, d7, sizeof(d7), 0xbc9f1d34},
           {12944174544826175378, d8, sizeof(d8), 0xbc9f1d34},
       }) {
    const char* data_batch[8];
    data_batch[0] = reinterpret_cast<const char*>(c.data);
    data_batch[1] = reinterpret_cast<const char*>(c.data);
    data_batch[2] = reinterpret_cast<const char*>(c.data);
    data_batch[3] = reinterpret_cast<const char*>(c.data);
    data_batch[4] = reinterpret_cast<const char*>(c.data);
    data_batch[5] = reinterpret_cast<const char*>(c.data);
    data_batch[6] = reinterpret_cast<const char*>(c.data);
    data_batch[7] = reinterpret_cast<const char*>(c.data);
    uint64_t h_value[8];
    Hash64Farm_Batch512(data_batch, &h_value[0], c.size);
    EXPECT_EQ(c.hash64, h_value[0]);
    EXPECT_EQ(c.hash64, h_value[1]);
    EXPECT_EQ(c.hash64, h_value[2]);
    EXPECT_EQ(c.hash64, h_value[3]);
    EXPECT_EQ(c.hash64, h_value[4]);
    EXPECT_EQ(c.hash64, h_value[5]);
    EXPECT_EQ(c.hash64, h_value[6]);
    EXPECT_EQ(c.hash64, h_value[7]);
  }
}
#endif

TEST(Hash, HashPtrIsNotIdentityFunction) {
  int* ptr = reinterpret_cast<int*>(0xcafe0000);
  EXPECT_NE(hash<int*>()(ptr), size_t{0xcafe0000});
}

static void BM_Hash32(int iters, int len) {
  std::string input(len, 'x');
  uint32 h = 0;
  for (int i = 0; i < iters; i++) {
    h = Hash32(input.data(), len, 1);
  }
  testing::BytesProcessed(static_cast<int64>(iters) * len);
  VLOG(1) << h;
}
BENCHMARK(BM_Hash32)->Range(1, 1024);

TEST(StringPieceHasher, Equality) {
  StringPieceHasher hasher;

  StringPiece s1("foo");
  StringPiece s2("bar");
  StringPiece s3("baz");
  StringPiece s4("zot");

  EXPECT_TRUE(hasher(s1) != hasher(s2));
  EXPECT_TRUE(hasher(s1) != hasher(s3));
  EXPECT_TRUE(hasher(s1) != hasher(s4));
  EXPECT_TRUE(hasher(s2) != hasher(s3));
  EXPECT_TRUE(hasher(s2) != hasher(s4));
  EXPECT_TRUE(hasher(s3) != hasher(s4));

  EXPECT_TRUE(hasher(s1) == hasher(s1));
  EXPECT_TRUE(hasher(s2) == hasher(s2));
  EXPECT_TRUE(hasher(s3) == hasher(s3));
  EXPECT_TRUE(hasher(s4) == hasher(s4));
}
static void BM_Hash64(int iters, int len) {
  std::string input(len, 'x');
  uint64 h = 0;
  for (int i = 0; i < iters; i++) {
    h = Hash64(input.data(), len, 1);
  }
  testing::BytesProcessed(static_cast<int64>(iters) * len);
  VLOG(1) << h;
}
BENCHMARK(BM_Hash64)->Range(48, 48);
BENCHMARK(BM_Hash64)->Range(47, 47);
BENCHMARK(BM_Hash64)->Range(1023, 1023);

static void BM_Hash64Farm(int iters, int len) {
  std::string input(len, 'x');
  uint64 h = 0;
  for (int i = 0; i < iters; i++) {
    h = ::util::Fingerprint64(input.data(), input.size());
  }
  testing::BytesProcessed(static_cast<int64>(iters) * len);
  VLOG(1) << h;
}
BENCHMARK(BM_Hash64Farm)->Range(8, 8);
BENCHMARK(BM_Hash64Farm)->Range(16, 16);
BENCHMARK(BM_Hash64Farm)->Range(32, 32);
BENCHMARK(BM_Hash64Farm)->Range(64, 64);
BENCHMARK(BM_Hash64Farm)->Range(128, 128);
BENCHMARK(BM_Hash64Farm)->Range(256, 256);
BENCHMARK(BM_Hash64Farm)->Range(512, 512);
BENCHMARK(BM_Hash64Farm)->Range(1024, 1024);

static void BM_Hash64BatchX4(int iters, int len) {
  std::string input_0(len, 'x');
  std::string input_1(len, 'x');
  std::string input_2(len, 'x');
  std::string input_3(len, 'x');
  uint64 h_0 = 0;
  uint64 h_1 = 0;
  uint64 h_2 = 0;
  uint64 h_3 = 0;
  for (int i = 0; i < iters; i++) {
    h_0 = Hash64(input_0.data(), len, 1);
    h_1 = Hash64(input_1.data(), len, 1);
    h_2 = Hash64(input_2.data(), len, 1);
    h_3 = Hash64(input_3.data(), len, 1);
  }
  testing::BytesProcessed(static_cast<int64>(iters) * len * 4);
  VLOG(1) << h_0;
}
BENCHMARK(BM_Hash64BatchX4)->Range(1, 1024);

static void BM_Hash64V3(int iters, int len) {
  std::string input(len, 'x');
  uint64 h = 0;
  for (int i = 0; i < iters; i++) {
    h = Hash64V3(input.data(), len, 1);
  }
  testing::BytesProcessed(static_cast<int64>(iters) * len);
  VLOG(1) << h;
}
BENCHMARK(BM_Hash64V3)->Range(1, 1024);

#if defined(__AVX512F__)
static void BM_Hash64BatchX8V3(int iters, int len) {
  uint64 h[8];
  std::string input_0(len, 'x');
  std::string input_1(len, 'x');
  std::string input_2(len, 'x');
  std::string input_3(len, 'x');
  std::string input_4(len, 'x');
  std::string input_5(len, 'x');
  std::string input_6(len, 'x');
  std::string input_7(len, 'x');
  const char* data_batch[8];
  for (int i = 0; i < iters; i++) {
    data_batch[0] = input_0.data(); 
    data_batch[1] = input_1.data(); 
    data_batch[2] = input_2.data(); 
    data_batch[3] = input_3.data(); 
    data_batch[4] = input_4.data(); 
    data_batch[5] = input_5.data(); 
    data_batch[6] = input_6.data(); 
    data_batch[7] = input_7.data(); 
    Hash64V3_Batch512(data_batch, h, len, 1);
  }
  testing::BytesProcessed(static_cast<int64>(iters) * len * 8);
  VLOG(1) << h[0];
}
BENCHMARK(BM_Hash64BatchX8V3)->Range(48, 48);
BENCHMARK(BM_Hash64BatchX8V3)->Range(47, 47);
BENCHMARK(BM_Hash64BatchX8V3)->Range(1023, 1023);

static void BM_Hash64BatchX8Farm(int iters, int len) {
  uint64_t h[8];
  std::string input_0(len, 'x');
  std::string input_1(len, 'x');
  std::string input_2(len, 'x');
  std::string input_3(len, 'x');
  std::string input_4(len, 'x');
  std::string input_5(len, 'x');
  std::string input_6(len, 'x');
  std::string input_7(len, 'x');
  const char* data_batch[8];
  for (int i = 0; i < iters; i++) {
    data_batch[0] = input_0.data(); 
    data_batch[1] = input_1.data(); 
    data_batch[2] = input_2.data(); 
    data_batch[3] = input_3.data(); 
    data_batch[4] = input_4.data(); 
    data_batch[5] = input_5.data(); 
    data_batch[6] = input_6.data(); 
    data_batch[7] = input_7.data(); 
    Hash64Farm_Batch512(data_batch, h, len);
  }
  testing::BytesProcessed(static_cast<int64>(iters) * len * 8);
  VLOG(1) << h[0];
}
BENCHMARK(BM_Hash64BatchX8Farm)->Range(8, 8);
BENCHMARK(BM_Hash64BatchX8Farm)->Range(16, 16);
BENCHMARK(BM_Hash64BatchX8Farm)->Range(32, 32);
BENCHMARK(BM_Hash64BatchX8Farm)->Range(64, 64);
BENCHMARK(BM_Hash64BatchX8Farm)->Range(128, 128);
BENCHMARK(BM_Hash64BatchX8Farm)->Range(256, 256);
BENCHMARK(BM_Hash64BatchX8Farm)->Range(512, 512);
BENCHMARK(BM_Hash64BatchX8Farm)->Range(1024, 1024);
#endif

TEST(StringPieceHasher, HashMap) {
  string s1("foo");
  string s2("bar");
  string s3("baz");

  StringPiece p1(s1);
  StringPiece p2(s2);
  StringPiece p3(s3);

  std::unordered_map<StringPiece, int, StringPieceHasher> map;

  map.insert(std::make_pair(p1, 0));
  map.insert(std::make_pair(p2, 1));
  map.insert(std::make_pair(p3, 2));
  EXPECT_EQ(map.size(), 3);

  bool found[3] = {false, false, false};
  for (auto const& val : map) {
    int x = val.second;
    EXPECT_TRUE(x >= 0 && x < 3);
    EXPECT_TRUE(!found[x]);
    found[x] = true;
  }
  EXPECT_EQ(found[0], true);
  EXPECT_EQ(found[1], true);
  EXPECT_EQ(found[2], true);

  auto new_iter = map.find("zot");
  EXPECT_TRUE(new_iter == map.end());

  new_iter = map.find("bar");
  EXPECT_TRUE(new_iter != map.end());

  map.erase(new_iter);
  EXPECT_EQ(map.size(), 2);

  found[0] = false;
  found[1] = false;
  found[2] = false;
  for (const auto& iter : map) {
    int x = iter.second;
    EXPECT_TRUE(x >= 0 && x < 3);
    EXPECT_TRUE(!found[x]);
    found[x] = true;
  }
  EXPECT_EQ(found[0], true);
  EXPECT_EQ(found[1], false);
  EXPECT_EQ(found[2], true);
}

}  // namespace tensorflow

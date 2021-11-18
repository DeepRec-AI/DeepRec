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

#include "tensorflow/core/framework/hash_table/bloom_filter_strategy.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(BloomFilterAdmitStrategy, Simple) {
  {
    // test for only one key
    int64 key = -123;
    BloomFilterAdmitStrategy bf(10, 1, DT_UINT8, {1, 1024});
    for (int64 k = 0; k < 9; ++k) {
      EXPECT_FALSE(bf.Admit(key));
    }
    for (int64 k = 0; k < 1000; ++k) {
      EXPECT_TRUE(bf.Admit(key));
    }
  }
  {
    // test multiple key
    std::vector<int64> keys = { 101, 202, -3003, -11111111, 99999999 };
    BloomFilterAdmitStrategy bf(10, 3, DT_UINT8, {1, 1024});
    for (int64 k = 0; k < 9; ++k) {
      for (auto& key : keys) {
        bf.Admit(key);
      }
    }
    for (int64 k = 0; k < 1000; ++k) {
      for (auto& key : keys) {
        EXPECT_TRUE(bf.Admit(key));
      }
    }
  }
  {
    // test admit with counting
    int64 key = 12345;
    BloomFilterAdmitStrategy bf(3, 3, DT_UINT8, {1, 1024});
    EXPECT_FALSE(bf.Admit(key, 2));
    EXPECT_TRUE(bf.Admit(key, 1));
    EXPECT_TRUE(bf.Admit(key, 100000));
    EXPECT_TRUE(bf.Admit(key));
  }
}

}  //namespace tensorflow

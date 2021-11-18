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

#ifndef TENSORFLOW_FRAMEWORK_HASH_TABLE_BLOOM_FILTER_STRATEGY_H_
#define TENSORFLOW_FRAMEWORK_HASH_TABLE_BLOOM_FILTER_STRATEGY_H_

#include "tensorflow/core/framework/hash_table/hash_table.h"

namespace tensorflow {

class BloomFilterAdmitStrategy : public HashTableAdmitStrategy {
 public:
  BloomFilterAdmitStrategy(int64 minimum_frequency,
                           int64 num_hash_func,
                           DataType dtype,
                           const TensorShape& shape,
                           int64 slice_offset = 0,
                           int64 max_slice_size = 1);
  virtual ~BloomFilterAdmitStrategy();
  bool Admit(int64 key) override;
  bool Admit(int64 key, int64 freq) override;
  std::vector<int8> Snapshot();
  void Restore(int64 src_beg, int64 src_length, int64 dst_beg,
               int64 dst_length, const std::vector<int8>& src);
 private:
  void GenerateSeeds();
  bool AdmitInternal(int64 key, int64 counting);
  uint64_t DefaultHashFunc(int64 key, int64 seed);
  int64 Read(int64 k) {
    int64 val = 0;
    switch (dtype_) {
      case DT_UINT8:
        val = *(reinterpret_cast<uint8*>(bucket_) + k);
        break;
      case DT_UINT16:
        val = *(reinterpret_cast<uint16*>(bucket_) + k);
        break;
      case DT_UINT32:
        val = *(reinterpret_cast<uint32*>(bucket_) + k);
        break;
      default:
        LOG(FATAL) << "Not support data type " << dtype_;
        break;
    }
    return val;
  }
  void Write(int64 val, int64 k)  {
    switch (dtype_) {
      case DT_UINT8:
        *(reinterpret_cast<uint8*>(bucket_) + k) = static_cast<uint8>(val);
        break;
      case DT_UINT16:
        *(reinterpret_cast<uint16*>(bucket_) + k) = static_cast<uint16>(val);
        break;
      case DT_UINT32:
        *(reinterpret_cast<uint32*>(bucket_) + k) = static_cast<uint32>(val);
        break;
      default:
        LOG(FATAL) << "Not support data type " << dtype_;
        break;
    }
  }
  int64 minimum_frequency_;
  int64 num_hash_func_;
  DataType dtype_;
  TensorShape shape_;
  int64 slice_offset_;
  int64 max_slice_size_;
  int8* bucket_;
  int64 segment_size_;
  std::vector<int64> seeds_;
  int64 max_freq_;
  mutex mu_;
  constexpr static int64 kDefaultFrequency = 1;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_HASH_TABLE_BLOOM_FILTER_STRATEGY_H_

/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================*/
#include "tensorflow/core/kernels/embedding_variable_test.h"

namespace tensorflow {
namespace embedding {
float PerfMemory(Tensor& default_value,
                const std::vector<int64>& id_list,
                int value_size, int64 default_value_dim,
                int64 filter_freq = 0) {
  auto ev = CreateEmbeddingVar(value_size, default_value,
                               default_value_dim, filter_freq);
  ValuePtr<float>* value_ptr = nullptr;
  bool is_filter = false;
  double start_mem, end_mem;
  start_mem = getResident() * getpagesize();
  for (int i = 0; i < id_list.size(); i++) {
    ev->LookupOrCreateKey(id_list[i], &value_ptr, &is_filter, false);
    if (is_filter)
      ev->flat(value_ptr, id_list[i]);
  }
  end_mem = getResident() * getpagesize();
  double used_mb = (end_mem - start_mem)/1000000;
  LOG(INFO)<<"[TestMemory]Use Memory: "<<used_mb;
  return used_mb;
}

TEST(EmbeddingVariabelMemoryTest, TestMemory) {
  int value_size = 32;
  int64 default_value_dim = 4096;
  int filter_freq = 2;
  Tensor default_value(
      DT_FLOAT, TensorShape({default_value_dim, value_size}));
  auto default_value_matrix = default_value.matrix<float>();
	for (int i = 0; i < default_value_dim; i++) {
		for (int j = 0 ; j < value_size; j++) {
			default_value_matrix(i, j) = i * value_size + j;
		}
	}

  int num_of_ids = 1000000;
  std::vector<int64> id_list(num_of_ids);
  for (int i = 0; i < num_of_ids; i++) {
    id_list[i] = i;
  }
  float used_mb = PerfMemory(default_value, id_list,
                             value_size, default_value_dim);
  float theoritical_mb =
      50 + num_of_ids * (32 + 32 + value_size * sizeof(float))/ 1000000;
  EXPECT_TRUE((used_mb > theoritical_mb * 0.99) &&
              (used_mb < theoritical_mb * 1.01));

  for (int i = 0; i < num_of_ids / 2; i++) {
    id_list.emplace_back(i);
  }
  used_mb = PerfMemory(default_value, id_list, value_size,
                       default_value_dim, filter_freq);
  theoritical_mb =
      50 + num_of_ids * (32 + 32 + 16 + value_size * sizeof(float)/2)/ 1000000;
  EXPECT_TRUE((used_mb > theoritical_mb * 0.99) &&
              (used_mb < theoritical_mb * 1.01));
}
} //namespace embedding
} //namespace tensorflow

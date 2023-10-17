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
void GenerateSkewIds(int num_of_ids, float skew_factor,
                     std::vector<int64>& hot_ids_list,
                     std::vector<int64>& cold_ids_list) {
  int num_of_hot_ids = num_of_ids * (1 - skew_factor);
	int num_of_cold_ids = num_of_ids - num_of_hot_ids;
	std::set<int64> hot_ids_set;
  std::set<int64> cold_ids_set;
  hot_ids_list.resize(num_of_hot_ids);
  cold_ids_list.resize(num_of_cold_ids);
  srand((unsigned)time(NULL));
  //Generate hot ids
  for (int i = 0; i < num_of_hot_ids; i++) {
    bool flag = false;
    int64 key;
    do {
      key = rand() % 100000000;
      flag = hot_ids_set.insert(key).second;
      hot_ids_list[i] = key;
    } while (!flag);
  }
  //Generate cold ids
  for (int i = 0; i < num_of_cold_ids; i++) {
    bool flag = false;
    int64 key;
    do {
      key = rand() % 100000000;
      if (hot_ids_set.find(key) != hot_ids_set.end()) {
        flag = false;
      } else {
        flag = cold_ids_set.insert(key).second;
        cold_ids_list[i] = key;
      }
    } while (!flag);
  }
}

void InitSkewInputBatch(std::vector<std::vector<int64>>& input_batches,
                        float skew_factor,
                        const std::vector<int64>& hot_ids_list,
                        const std::vector<int64>& cold_ids_list) {
  srand((unsigned)time(NULL));
  int num_of_hot_ids = hot_ids_list.size();
  int num_of_cold_ids = cold_ids_list.size();
  int num_of_batch = input_batches.size();
  for (int i = 0; i < input_batches.size(); i++) {
    for (int j = 0; j < input_batches[i].size(); j++) {
      int tmp = rand() % 10;
      if ((float)tmp * 0.1 < skew_factor) {
        int pos = rand() % num_of_hot_ids;
        input_batches[i][j] = hot_ids_list[pos];
      } else {
        int pos = rand() % num_of_cold_ids;
        input_batches[i][j] = cold_ids_list[pos];
      }
    }
  }
}


void GenerateSkewInput(int num_of_ids, float skew_factor,
                      std::vector<std::vector<int64>>& input_batches) {
  std::vector<int64> hot_ids_list;
  std::vector<int64> cold_ids_list;
  //Generate hot ids
  GenerateSkewIds(num_of_ids, skew_factor,
                  hot_ids_list, cold_ids_list);
  //Select id for each batch
  InitSkewInputBatch(input_batches, skew_factor,
                     hot_ids_list, cold_ids_list);
}

void thread_lookup_or_create(
    EmbeddingVar<int64, float>* ev,
    const int64* input_batch,
    float* default_value,
    int default_value_dim,
    float** outputs, int value_size,
    int start, int end) {
  void* value_ptr = nullptr;
	bool is_filter = false;
  for (int i = start; i < end; i++) {
    ev->LookupOrCreateKey(input_batch[i], &value_ptr, &is_filter, false);
    if (is_filter) {
      auto val = ev->flat(value_ptr);
      memcpy(outputs[i], &val(0), sizeof(float) * value_size);
    } else {
      int default_value_index = input_batch[i] % default_value_dim;
      memcpy(outputs[i], default_value + default_value_index * value_size, sizeof(float) * value_size);
    }
  }
}

double PerfLookupOrCreate(
    const std::vector<std::vector<int64>>& input_batches,
    int num_thread, int filter_freq = 0) {
  int value_size = 32;
  int64 default_value_dim = 4096;
  Tensor default_value(DT_FLOAT, TensorShape({default_value_dim, value_size}));
  auto default_value_matrix = default_value.matrix<float>();
	for (int i = 0; i < default_value_dim; i++) {
		for (int j = 0 ; j < value_size; j++) {
			default_value_matrix(i, j) = i * value_size + j;
		}
	}
  auto ev = CreateEmbeddingVar(value_size, default_value,
                               default_value_dim, filter_freq);
  std::vector<std::thread> worker_threads(num_thread);
  double total_time = 0.0;
  timespec start, end;
  for (int k = 0; k < input_batches.size(); k++) {
    //Allocate Outputs for each batch
    std::vector<float*> outputs(input_batches[k].size());
    for (int i = 0; i < outputs.size(); i++) {
      outputs[i] =
          (float*)cpu_allocator()->AllocateRaw(0, sizeof(float) * value_size);
    }
    //Execution
    std::vector<std::pair<int, int>> thread_task_range(num_thread);
    for (int i = 0; i < num_thread; i++) {
      int st = input_batches[k].size() / num_thread * i;
      int ed = input_batches[k].size() / num_thread * (i + 1);
      ed = (ed > input_batches[k].size()) ? input_batches[k].size() : ed;
      thread_task_range[i].first = st;
      thread_task_range[i].second = ed;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < num_thread; i++) {
      worker_threads[i] = std::thread(thread_lookup_or_create,
                                      ev, input_batches[k].data(),
                                      default_value_matrix.data(),
                                      default_value_dim,
                                      outputs.data(), value_size,
                                      thread_task_range[i].first,
                                      thread_task_range[i].second);
    }
    for (int i = 0; i < num_thread; i++) {
      worker_threads[i].join();
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (k > 10)
      total_time += ((double)(end.tv_sec - start.tv_sec) *
                     1000000000 + end.tv_nsec - start.tv_nsec);
    //Check
    for (int i = 0; i < input_batches[k].size(); i++) {
      int64 key =  input_batches[k][i];
      float* output = outputs[i];
      for (int j = 0; j < value_size; j++) {
        float val = default_value_matrix(key % default_value_dim, j);
        if (output[j] != val) {
          LOG(INFO)<<"Value Error: outputs["<<key<<"]["<<j
                    <<"] is "<<output[j]<<", while the anwser is "<<val;
          return -1.0;
        }
      }
    }
    //Deallocate Output
    for (auto ptr: outputs) {
      cpu_allocator()->DeallocateRaw(ptr);
    }
  }
  ev->Unref();
  return total_time;
}

TEST(EmbeddingVariablePerformanceTest, TestLookupOrCreate) {
  int num_of_batch = 100;
  int batch_size = 1024 * 128;
  int num_of_ids = 5000000;
  std::vector<std::vector<int64>> input_batches(num_of_batch);
  for (int i = 0; i < num_of_batch; i++) {
    input_batches[i].resize(batch_size);
  }
  LOG(INFO)<<"[TestLookupOrCreate] Start generating skew input";
  GenerateSkewInput(num_of_ids, 0.8, input_batches);
  LOG(INFO)<<"[TestLookupOrCreate] Finish generating skew input";
  std::vector<int> num_thread_vec({1, 2, 4, 8, 16});
  for (auto num_thread: num_thread_vec) {
    LOG(INFO)<<"[TestLookupOrCreate] Test LookupOrCreate With "
             <<num_thread<<" threads.";
    double exec_time = PerfLookupOrCreate(input_batches, num_thread);
    if (exec_time == -1.0) {
      LOG(INFO)<<"[TestLookupOrCreate] Test Failed";
    } else {
      LOG(INFO)<<"[TestLookupOrCreate] Performance of LookupOrCreate With "
               <<num_thread<<" threads: "<<exec_time/1000000<<" ms";
    }
  }
}

void thread_lookup(
    EmbeddingVar<int64, float>* ev,
    const int64* input_batch,
    float** outputs, int value_size,
    int start, int end) {
  void* value_ptr = nullptr;
	bool is_filter = false;
  for (int i = start; i < end; i++) {
    ev->LookupKey(input_batch[i], &value_ptr);
    auto val = ev->flat(value_ptr);
    memcpy(outputs[i], &val(0), sizeof(float) * value_size);
  }
}

double PerfLookup(
    EmbeddingVar<int64, float>* ev,
    const std::vector<std::vector<int64>>& input_batches,
    int num_thread,
    int value_size, float* default_value,
    int64 default_value_dim) {
  std::vector<std::thread> worker_threads(num_thread);
  double total_time = 0.0;
  timespec start, end;
  for (int k = 0; k < input_batches.size(); k++) {
    //Allocate Outputs for each batch
    std::vector<float*> outputs(input_batches[k].size());
    for (int i = 0; i < outputs.size(); i++) {
      outputs[i] =
          (float*)cpu_allocator()->AllocateRaw(0, sizeof(float) * value_size);
    }
    //Execution
    std::vector<std::pair<int, int>> thread_task_range(num_thread);
    for (int i = 0; i < num_thread; i++) {
      int st = input_batches[k].size() / num_thread * i;
      int ed = input_batches[k].size() / num_thread * (i + 1);
      ed = (ed > input_batches[k].size()) ? input_batches[k].size() : ed;
      thread_task_range[i].first = st;
      thread_task_range[i].second = ed;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < num_thread; i++) {
      worker_threads[i] = std::thread(thread_lookup,
                                      ev, input_batches[k].data(),
                                      outputs.data(), value_size,
                                      thread_task_range[i].first,
                                      thread_task_range[i].second);
    }
    for (int i = 0; i < num_thread; i++) {
      worker_threads[i].join();
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (k > 10)
      total_time += ((double)(end.tv_sec - start.tv_sec) *
                     1000000000 + end.tv_nsec - start.tv_nsec);
    //Check
    for (int i = 0; i < input_batches[k].size(); i++) {
      int64 key =  input_batches[k][i];
      float* output = outputs[i];
      for (int j = 0; j < value_size; j++) {
        float val = default_value[(key % default_value_dim) * value_size + j];
        if (output[j] != val) {
          LOG(INFO)<<"Value Error: outputs["<<key<<"]["<<j
                    <<"] is "<<output[j]<<", while is the anwser is "<<val;
          return -1.0;
        }
      }
    }
    //Deallocate Output
    for (auto ptr: outputs) {
      cpu_allocator()->DeallocateRaw(ptr);
    }
  }
  return total_time;
}

TEST(EmbeddingVariablePerformanceTest, TestLookup) {
  int num_of_batch = 100;
  int batch_size = 1024 * 128;
  int num_of_ids = 5000000;
  int value_size = 32;
  int64 default_value_dim = 4096;
  float skew_factor = 0.8;

  LOG(INFO)<<"[TestLookup] Start initializing EV storage.";
  std::vector<int64> hot_ids_list;
  std::vector<int64> cold_ids_list;
  GenerateSkewIds(num_of_ids, skew_factor, hot_ids_list, cold_ids_list);

  Tensor default_value(
      DT_FLOAT, TensorShape({default_value_dim, value_size}));
  auto default_value_matrix = default_value.matrix<float>();
	for (int i = 0; i < default_value_dim; i++) {
		for (int j = 0 ; j < value_size; j++) {
			default_value_matrix(i, j) = i * value_size + j;
		}
	}
  auto ev = CreateEmbeddingVar(value_size, default_value, default_value_dim);
  void* value_ptr = nullptr;
  bool is_filter = false;
  for (int i = 0; i < hot_ids_list.size(); i++) {
    ev->LookupOrCreateKey(hot_ids_list[i], &value_ptr, &is_filter, false);
  }
  for (int i = 0; i < cold_ids_list.size(); i++) {
    ev->LookupOrCreateKey(cold_ids_list[i], &value_ptr, &is_filter, false);
  }
  LOG(INFO)<<"[TestLookup] End initializing EV storage.";

  LOG(INFO)<<"[TestLookup] Start generating skew input";
  std::vector<std::vector<int64>> input_batches(num_of_batch);
  for (int i = 0; i < num_of_batch; i++) {
    input_batches[i].resize(batch_size);
  }
  InitSkewInputBatch(input_batches, skew_factor, hot_ids_list, cold_ids_list);
  LOG(INFO)<<"[TestLookup] Finish generating skew input";
  std::vector<int> num_thread_vec({1, 2, 4, 8, 16});
  for (auto num_thread: num_thread_vec) {
    LOG(INFO)<<"[TestLookup] Test Lookup With "<<num_thread<<" threads.";
    double exec_time = PerfLookup(ev, input_batches, num_thread,
                                  value_size, (float*)default_value.data(),
                                  default_value_dim);
    if (exec_time == -1.0) {
      LOG(INFO)<<"[TestLookup] Test Failed";
    } else {
      LOG(INFO)<<"[TestLookup] Performance of Lookup With "
               <<num_thread<<" threads: "<<exec_time/1000000<<" ms";
    }
  }
  ev->Unref();
}

string Prefix(const string& prefix) {
  return strings::StrCat(testing::TmpDir(), "/", prefix);
}

void PerfSave(Tensor& default_value,
              const std::vector<int64>& id_list,
              int value_size, int64 default_value_dim,
              int64 steps_to_live = 0,
              float l2_weight_threshold = -1.0) {
  auto ev = CreateEmbeddingVar(
      value_size, default_value,
      default_value_dim, 0, steps_to_live,
      l2_weight_threshold);
  void* value_ptr = nullptr;
  bool is_filter = false;
  srand((unsigned)time(NULL));

  for (int i = 0; i < id_list.size(); i++) {
    ev->LookupOrCreateKey(id_list[i], &value_ptr, &is_filter, false);
    ev->flat(value_ptr);
    int64 global_step = rand() % 100;
    ev->UpdateVersion(value_ptr, global_step);
  }
  Tensor part_offset_tensor(DT_INT32,  TensorShape({kSavedPartitionNum + 1}));

  BundleWriter writer(Env::Default(), Prefix("foo"));
  timespec start, end;
  double total_time = 0.0;
  embedding::ShrinkArgs shrink_args;
  shrink_args.global_step = 100;
  clock_gettime(CLOCK_MONOTONIC, &start);
  ev->Save("var", Prefix("foo"), &writer, shrink_args);
  clock_gettime(CLOCK_MONOTONIC, &end);
  total_time += (double)(end.tv_sec - start.tv_sec) *
                 1000000000 + end.tv_nsec - start.tv_nsec;
  TF_ASSERT_OK(writer.Finish());
  LOG(INFO)<<"[TestSave]execution time: "
           << total_time/1000000 <<"ms";
  ev->Unref();
}

TEST(EmbeddingVariablePerformanceTest, TestSave) {
  int value_size = 32;
  int64 default_value_dim = 4096;
  Tensor default_value(
      DT_FLOAT, TensorShape({default_value_dim, value_size}));
  auto default_value_matrix = default_value.matrix<float>();
	for (int i = 0; i < default_value_dim; i++) {
		for (int j = 0 ; j < value_size; j++) {
			default_value_matrix(i, j) = i * value_size + j;
		}
	}

  int num_of_ids = 1000000;
  srand((unsigned)time(NULL));
  std::vector<int64> id_list(num_of_ids);
  for (int i = 0; i < num_of_ids; i++) {
    id_list[i] = rand() % 50000000;
  }
  PerfSave(default_value, id_list, value_size, default_value_dim);
}

TEST(EmbeddingVariablePerformanceTest, TestGlobalStepEviction) {
  int value_size = 32;
  int64 default_value_dim = 4096;
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
  srand((unsigned)time(NULL));
  for (int i = 0; i < num_of_ids; i++) {
    id_list[i] = rand() % 50000000;
  }
  PerfSave(default_value, id_list, value_size, default_value_dim, 80);
}

TEST(EmbeddingVariablePerformanceTest, TestL2WeightEviction) {
  int value_size = 32;
  int64 default_value_dim = 4096;
  Tensor default_value(
      DT_FLOAT, TensorShape({default_value_dim, value_size}));
  auto default_value_matrix = default_value.matrix<float>();
	for (int i = 0; i < default_value_dim; i++) {
		for (int j = 0 ; j < value_size; j++) {
			default_value_matrix(i, j) = i * value_size + j;
		}
	}

  int l2_weight_threshold_index = default_value_dim * 0.2;
  float l2_weight_threshold = 0.0;
  for (int64 j = 0; j < value_size; j++) {
    l2_weight_threshold +=
        pow(default_value_matrix(l2_weight_threshold_index, j), 2);
  }
  l2_weight_threshold *= 0.5;

  int num_of_ids = 1000000;
  std::vector<int64> id_list(num_of_ids);
  srand((unsigned)time(NULL));
  for (int i = 0; i < num_of_ids; i++) {
    id_list[i] = rand() % 50000000;
  }
  PerfSave(default_value, id_list, value_size,
           default_value_dim, 0, l2_weight_threshold);
}

TEST(EmbeddingVariablePerformaceTest, TestCounterFilterLookupOrCreate) {
  int num_of_batch = 100;
  int batch_size = 1024 * 128;
  int num_of_ids = 5000000;
  int64 filter_freq = 5;
  std::vector<std::vector<int64>> input_batches(num_of_batch);
  for (int i = 0; i < num_of_batch; i++) {
    input_batches[i].resize(batch_size);
  }
  LOG(INFO)<<"[TestCounterFilterLookupOrCreate] Start generating skew input";
  GenerateSkewInput(num_of_ids, 0.8, input_batches);
  LOG(INFO)<<"[TestCounterFilterLookupOrCreate] Finish generating skew input";
  std::vector<int> num_thread_vec({1, 2, 4, 8, 16});
  for (auto num_thread: num_thread_vec) {
    LOG(INFO)<<"[TestCounterFilterLookupOrCreate] Test LookupOrCreate With "
             <<num_thread<<" threads.";
    double exec_time = PerfLookupOrCreate(input_batches, num_thread, filter_freq);
    if (exec_time == -1.0) {
      LOG(INFO)<<"[TestCounterFilterLookupOrCreate] Test Failed";
    } else {
      LOG(INFO)<<"[TestCounterFilterLookupOrCreate] Performance of LookupOrCreate With "
               <<num_thread<<" threads: "<<exec_time/1000000<<" ms";
    }
  }
}
} //namespace embedding
} //namespace tensorflow

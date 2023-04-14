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
=======================================================================*/

#define EIGEN_USE_THREADS

#include <immintrin.h>

#include <unordered_map>

#include "sparsehash/dense_hash_map"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/task_runner.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/work_sharder.h"
namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace {

const char* kInferenceMode = "INFERENCE_MODE";

enum Combiner { Mean, Sum, Sqrtn };

void ParallelFor(const std::function<void(size_t)>& f, size_t n,
                 thread::ThreadPool* thread_pool) {
  if (n == 0) return;
  if (thread_pool == nullptr) {
    for (size_t i = 0; i < n; ++i) {
      f(i);
    }
  } else {
    BlockingCounter counter(n - 1);
    for (size_t i = 1; i < n; ++i) {
      thread_pool->Schedule([i, &f, &counter] {
        f(i);
        counter.DecrementCount();
      });
    }
    f(0);
    counter.Wait();
  }
}

}  // namespace

template <typename TKey, typename TValue>
class GroupEmbeddingVariableLookupCpuOp : public OpKernel {
 public:
  explicit GroupEmbeddingVariableLookupCpuOp(OpKernelConstruction* c)
      : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    // OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(c, c->GetAttr("is_use_default_value_tensor",
                                 &is_use_default_value_tensor_));
    OP_REQUIRES_OK(c, c->GetAttr("is_inference", &is_inference_));
    OP_REQUIRES_OK(c, c->GetAttr("ignore_weights", &ignore_weights_));
    bool is_inference;
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference));
    is_inference_ |= is_inference;
    thread_pool_ = std::make_unique<thread::ThreadPool>(
        Env::Default(), "GroupEmbedding", num_lookups_);
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                             int64 total_dim,
                             int64 len) { return default_v + len * index; };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                             int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim);
      };
    }
    if (!is_inference_) {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key, TValue* val,
                      TValue* default_v, int count) {
        ev->LookupOrCreate(key, val, default_v, count);
        return Status::OK();
      };
    } else {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key, TValue* val,
                      TValue* default_v, int count) {
        ev->LookupOrCreate(key, val, default_v);
        return Status::OK();
      };
    }
  }

  void Compute(OpKernelContext* ctx) override {
    /*
      step 1: unique and assign unique output and index
      step 2: doing unique value gather
      step 3: assign unique embedding to batch result and pooling
    */
    const Tensor& dense_shape_tensor = ctx->input(num_lookups_ * 4);
    auto dense_shape = dense_shape_tensor.flat<int>().data();
    int batch_size = dense_shape[0];
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    // ScopedPerThreadMaxParallelism(num_lookups_ * 2);
    // uint64_t op_start_micros = Env::Default()->NowMicros();
    auto do_compute = [this, ctx, batch_size, worker_threads](int i) {
      EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, i), &embedding_var));
      core::ScopedUnref unref_me(embedding_var);

      const Tensor& sp_values_tensor = ctx->input(num_lookups_ + i);
      auto sp_values = sp_values_tensor.flat<TKey>().data();
      const Tensor& sp_indices_tensor = ctx->input(num_lookups_ * 2 + i);
      auto sp_indices = sp_indices_tensor.flat<int64>().data();
      int nnz = sp_indices_tensor.shape().dim_size(0);

      OP_REQUIRES(
          ctx,
          !embedding_var->IsMultiLevel() || (embedding_var->IsMultiLevel() &&
                                             embedding_var->CacheSize() >= nnz),
          errors::InvalidArgument("MultiLevel EV's Cache size ",
                                  embedding_var->CacheSize(),
                                  " should large than IDs in batch ", nnz));

      TensorShape unique_idx_tensor_shape =
          TensorShape(std::vector<int64>({static_cast<long long>(nnz)}));
      Tensor* unique_idx_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2 * num_lookups_ + i,
                                               unique_idx_tensor_shape,
                                               &unique_idx_tensor));
      auto unique_idx = unique_idx_tensor->flat<int>().data();

      TensorShape batch_nums_tensor_shape =
          TensorShape(std::vector<int64>({static_cast<long long>(batch_size)}));
      Tensor* batch_nums_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(3 * num_lookups_ + i,
                                               batch_nums_tensor_shape,
                                               &batch_nums_tensor));
      auto batch_nums = batch_nums_tensor->flat<int>().data();

      // Stage 1
      google::dense_hash_map<TKey, int32> unique_map;
      unique_map.set_empty_key(std::numeric_limits<TKey>::max());
      unique_map.resize(2 * nnz);
      int global_batch_num = 0;
      int batch_id = 0;

      for (int64 k = 0, j = 0; k < nnz; ++k) {
        /***********Doing Unique **************/
        auto id = sp_values[k];
        int new_batch_id = sp_indices[k];
        if (new_batch_id != batch_id) {
          batch_nums[batch_id] = global_batch_num;
          batch_id = new_batch_id;
        }
        global_batch_num++;
        auto it = unique_map.emplace(id, j);
        unique_idx[k] = it.first->second;

        if (it.second) {
          ++j;
        }
      }
      // process final batch
      batch_nums[batch_id] = global_batch_num;

      int unique_nnz = unique_map.size();
      Tensor* unique_tensor = nullptr;
      TensorShape unique_shape{static_cast<int64>(unique_nnz)};
      OP_REQUIRES_OK(ctx, ctx->allocate_output(num_lookups_ + i, unique_shape,
                                               &unique_tensor));
      auto* unique = unique_tensor->flat<TKey>().data();

      for (auto it : unique_map) {
        unique[it.second] = it.first;
      }

      TValue* default_v = nullptr;
      if (is_use_default_value_tensor_) {
        default_v =
            reinterpret_cast<TValue*>(ctx->input(num_lookups_ * 4 + 1).data());
      } else {
        default_v = embedding_var->GetDefaultValuePtr();
      }

      // Stage 2
      Tensor unique_embedding;
      unique_shape.AppendShape({static_cast<int64>(dimension_)});
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<TValue>::v(), unique_shape,
                                  &unique_embedding, attr));
      auto unique_embedding_data = unique_embedding.flat<TValue>().data();

      int slice_bytes = unique_nnz * dimension_ * 1000;
      auto do_lookup = [this, ctx, embedding_var, unique, default_v,
                        unique_embedding_data](int64 start, int64 end) {
        for (int k = start; k < end; ++k) {
          TValue* default_v_ptr = get_default_v_fn_(
              default_v, unique[k], k, embedding_var->GetDefaultValueDim(),
              embedding_var->ValueLen());
          OP_REQUIRES_OK(ctx, lookup_fn_(embedding_var, unique[k],
                                         unique_embedding_data + k * dimension_,
                                         default_v_ptr, 1/*count*/));
        }
      };
      Shard(worker_threads->num_threads, worker_threads->workers, unique_nnz,
            slice_bytes /*cost*/, do_lookup);  // Parallel on batch

      if (embedding_var->IsMultiLevel()) {
        embedding::BatchCache<TKey>* cache = embedding_var->Cache();
        embedding_var->storage_manager()->Schedule(
            [embedding_var, sp_values_tensor] {
              embedding::BatchCache<TKey>* cache = embedding_var->Cache();
              cache->add_to_rank(sp_values_tensor);
            });
      }

      std::vector<TValue> default_weights(nnz, 1.0);
      TValue* sp_weights = default_weights.data();
      // TValue* sp_weights = default_v;
      if (!ignore_weights_) {
        const Tensor& sp_weights_tensor =
            ctx->input(this->num_lookups_ * 3 + i);
        sp_weights =
            const_cast<TValue*>(sp_weights_tensor.flat<TValue>().data());
      }

      // Stage 3
      TensorShape emb_vectors_tensor_shape = TensorShape(
          std::vector<int64>({static_cast<long long>(batch_size), dimension_}));
      Tensor* gather_embedding_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &gather_embedding_tensor));
      auto gather_embedding = gather_embedding_tensor->flat<TValue>().data();

      slice_bytes = nnz / batch_size * dimension_ * 1000;
      if (combiner_ == "mean") {
        auto embedding_var_mean_combiner =
            [this, &gather_embedding, batch_nums, unique_idx, unique,
             unique_embedding_data, sp_weights](int64 start, int64 end) {
              for (int64 i = start; i < end; ++i) {
                std::vector<TValue> tmp_embedding(dimension_, 0.0f);
                int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
                int batch_num = batch_nums[i] - batch_offset;
                for (int j = 0; j < batch_num; ++j) {
                  int unique_indice = unique_idx[batch_offset + j];
                  float* u_embedding =
                      unique_embedding_data + unique_indice * dimension_;
                  for (int d = 0; d < dimension_; ++d) {
                    tmp_embedding[d] = std::fma(*(u_embedding + d),
                                                sp_weights[batch_offset + j],
                                                tmp_embedding[d]);
                  }
                }
                for (int d = 0; d < dimension_; ++d) {
                  gather_embedding[i * dimension_ + d] =
                      tmp_embedding[d] / batch_num;
                }
              }
            };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/,
              embedding_var_mean_combiner);  // Parallel on batch
      } else if (combiner_ == "sum") {
        auto embedding_var_sum_combiner =
            [this, &gather_embedding, batch_nums, unique_idx, unique,
             unique_embedding_data, sp_weights](int64 start, int64 end) {
              for (int64 i = start; i < end; ++i) {
                std::vector<TValue> tmp_embedding(dimension_, 0.0f);
                int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
                int batch_num = batch_nums[i] - batch_offset;
                for (int j = 0; j < batch_num; ++j) {
                  int unique_indice = unique_idx[batch_offset + j];
                  float* u_embedding =
                      unique_embedding_data + unique_indice * dimension_;
                  for (int d = 0; d < dimension_; ++d) {
                    tmp_embedding[d] = std::fma(u_embedding[d],
                                                sp_weights[batch_offset + j],
                                                tmp_embedding[d]);
                  }
                }
                memcpy(gather_embedding + i * dimension_, tmp_embedding.data(),
                       sizeof(float) * dimension_);
              }
            };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/,
              embedding_var_sum_combiner);  // Parallel on batch
      } else {
        auto embedding_var_sqrtn_combiner =
            [this, &gather_embedding, batch_nums, unique_idx, unique,
             unique_embedding_data, sp_weights](int64 start, int64 end) {
              for (int64 i = start; i < end; ++i) {
                std::vector<TValue> tmp_embedding(dimension_, 0.0f);
                int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
                int batch_num = batch_nums[i] - batch_offset;
                for (int j = 0; j < batch_num; ++j) {
                  int unique_indice = unique_idx[batch_offset + j];
                  float* u_embedding =
                      unique_embedding_data + unique_indice * dimension_;
                  for (int d = 0; d < dimension_; ++d) {
                    tmp_embedding[d] = std::fma(u_embedding[d],
                                                sp_weights[batch_offset + j],
                                                tmp_embedding[d]);
                  }
                }
                for (int d = 0; d < dimension_; ++d) {
                  gather_embedding[i * dimension_ + d] =
                      tmp_embedding[d] / sqrtf(batch_num);
                }
              }
            };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/,
              embedding_var_sqrtn_combiner);  // Parallel on batch
      }
      // LOG(INFO) << "idx " << i << " embedding time stats: "
      //           << " unique_secs: " <<
      //           strings::HumanReadableElapsedTime(unique_secs)
      //           << " lookup_secs: " <<
      //           strings::HumanReadableElapsedTime(lookup_secs)
      //           << " pooling_secs: " <<
      //           strings::HumanReadableElapsedTime(pooling_secs);
    };
    ParallelFor(do_compute, num_lookups_, thread_pool_.get());
  }

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::function<TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
  std::function<Status(EmbeddingVar<TKey, TValue>* ev, TKey key, TValue* val,
                       TValue* default_v, int count)>
      lookup_fn_;
  std::string combiner_;
  // float max_norm_;
  int num_lookups_;
  int dimension_;
  bool is_use_default_value_tensor_;
  bool ignore_weights_;
  bool is_inference_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type) \
  REGISTER_KERNEL_BUILDER(                         \
      Name("GroupEmbeddingVarLookup")              \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<key_type>("Tkeys")       \
          .TypeConstraint<value_type>("dtype"),    \
      GroupEmbeddingVariableLookupCpuOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

template <typename TKey, typename TValue>
class GroupVariableLookupCpuOp : public OpKernel {
 public:
  explicit GroupVariableLookupCpuOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    // OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    thread_pool_ = std::make_unique<thread::ThreadPool>(
        Env::Default(), "GroupEmbedding", num_lookups_ / 2);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& dense_shape_tensor = ctx->input(num_lookups_ * 4);
    auto dense_shape = dense_shape_tensor.flat<int>().data();
    int batch_size = dense_shape[0];
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    auto do_compute = [this, ctx, batch_size, worker_threads](int i) {
      const Tensor& emb_variable_tensor = ctx->input(i);
      const Tensor& sp_values_tensor = ctx->input(num_lookups_ + i);
      // int64 emb_vec_size = emb_variable_tensor.shape().dim_size(1);
      auto embedding_variable = emb_variable_tensor.flat<TValue>().data();

      auto sp_values = sp_values_tensor.flat<TKey>().data();
      const Tensor& sp_indices_tensor = ctx->input(num_lookups_ * 2 + i);
      auto sp_indices = sp_indices_tensor.flat<int64>().data();
      int nnz = sp_indices_tensor.shape().dim_size(0);

      TensorShape emb_vectors_tensor_shape = TensorShape(
          std::vector<int64>({static_cast<long long>(batch_size), dimension_}));
      Tensor* emb_vectors_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &emb_vectors_tensor));
      auto emb_vectors = emb_vectors_tensor->flat<TValue>().data();

      /***********Filling Unique Idx **************/
      TensorShape unique_idx_tensor_shape =
          TensorShape(std::vector<int64>({static_cast<long long>(nnz)}));
      Tensor* unique_idx_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2 * num_lookups_ + i,
                                               unique_idx_tensor_shape,
                                               &unique_idx_tensor));
      auto unique_idx = unique_idx_tensor->flat<int>().data();

      TensorShape batch_nums_tensor_shape =
          TensorShape(std::vector<int64>({static_cast<long long>(batch_size)}));
      Tensor* batch_nums_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(3 * num_lookups_ + i,
                                               batch_nums_tensor_shape,
                                               &batch_nums_tensor));
      auto batch_nums = batch_nums_tensor->flat<int>().data();
      // do unique
      google::dense_hash_map<TKey, int32> unique_map;
      unique_map.set_empty_key(std::numeric_limits<TKey>::max());
      unique_map.resize(2 * nnz);
      int global_batch_num = 0;
      int batch_id = 0;

      for (int64 k = 0, j = 0; k < nnz; ++k) {
        auto id = sp_values[k];
        int new_batch_id = sp_indices[k];
        if (new_batch_id != batch_id) {
          batch_nums[batch_id] = global_batch_num;
          batch_id = new_batch_id;
        }
        global_batch_num++;
        auto it = unique_map.emplace(id, j);
        unique_idx[k] = it.first->second;

        if (it.second) {
          ++j;
        }
      }
      batch_nums[batch_id] = global_batch_num;

      /***********Filling Unique Value **************/
      Tensor* unique_tensor = nullptr;
      TensorShape unique_shape{static_cast<int64>(unique_map.size())};
      OP_REQUIRES_OK(ctx, ctx->allocate_output(num_lookups_ + i, unique_shape,
                                               &unique_tensor));
      auto* unique = unique_tensor->flat<TKey>().data();

      for (auto it : unique_map) {
        unique[it.second] = it.first;
      }

      std::vector<TValue> default_weights(nnz, 1.0);
      TValue* sp_weights = default_weights.data();
      if (!ignore_weights_) {
        const Tensor& sp_weights_tensor =
            ctx->input(this->num_lookups_ * 3 + i);
        sp_weights =
            const_cast<TValue*>(sp_weights_tensor.flat<TValue>().data());
      }
      int slice_bytes = nnz / batch_size * dimension_ * 1000;
      if (combiner_ == "mean") {
        auto do_var_mean = [this, &emb_vectors, batch_nums, unique_idx, unique,
                            sp_weights,
                            embedding_variable](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
            std::vector<TValue> tmp_embedding(dimension_, 0.0f);
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              int unique_id = unique[unique_indice];
              for (int d = 0; d < dimension_; ++d) {
                tmp_embedding[d] =
                    std::fma(embedding_variable[unique_id * dimension_ + d],
                             sp_weights[batch_offset + j], tmp_embedding[d]);
              }
            }
            for (int d = 0; d < dimension_; ++d) {
              emb_vectors[i * dimension_ + d] = tmp_embedding[d] / batch_num;
            }
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/,
              do_var_mean);  // Parallel on batch
      } else if (combiner_ == "sum") {
        auto do_var_sum = [this, &emb_vectors, batch_nums, unique_idx, unique,
                           sp_weights,
                           embedding_variable](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
            std::vector<TValue> tmp_embedding(dimension_, 0.0f);
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              int unique_id = unique[unique_indice];
              for (int d = 0; d < dimension_; ++d) {
                tmp_embedding[d] =
                    std::fma(embedding_variable[unique_id * dimension_ + d],
                             sp_weights[batch_offset + j], tmp_embedding[d]);
              }
            }
            memcpy(emb_vectors + i * dimension_, tmp_embedding.data(),
                   sizeof(float) * dimension_);
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/,
              do_var_sum);  // Parallel on batch
      } else {
        auto do_var_sqrtn = [this, &emb_vectors, batch_nums, unique_idx, unique,
                             sp_weights,
                             embedding_variable](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
            std::vector<TValue> tmp_embedding(dimension_, 0.0f);
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              int unique_id = unique[unique_indice];
              for (int d = 0; d < dimension_; ++d) {
                tmp_embedding[d] =
                    std::fma(embedding_variable[unique_id * dimension_ + d],
                             sp_weights[batch_offset + j], tmp_embedding[d]);
              }
            }
            for (int d = 0; d < dimension_; ++d) {
              emb_vectors[i * dimension_ + d] =
                  tmp_embedding[d] / sqrtf(batch_num);
            }
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/,
              do_var_sqrtn);  // Parallel on batch
      }
    };

    ParallelFor(do_compute, num_lookups_, thread_pool_.get());
  }

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::string combiner_;
  // float max_norm_;
  int num_lookups_;
  int dimension_;
  bool ignore_weights_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type)                  \
  REGISTER_KERNEL_BUILDER(Name("GroupVariableLookup")               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<key_type>("Tkeys")    \
                              .TypeConstraint<value_type>("dtype"), \
                          GroupVariableLookupCpuOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

template <typename TKey, typename TValue>
class GroupEmbeddingVariableLookupDenseCpuOp : public OpKernel {
 public:
  explicit GroupEmbeddingVariableLookupDenseCpuOp(OpKernelConstruction* c)
      : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    // OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(c, c->GetAttr("is_use_default_value_tensor",
                                 &is_use_default_value_tensor_));
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference_));

    thread_pool_ = std::make_unique<thread::ThreadPool>(
        Env::Default(), "GroupEmbeddingDense", num_lookups_ / 2);
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                             int64 total_dim,
                             int64 len) { return default_v + len * index; };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                             int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim);
      };
    }
    if (!is_inference_) {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key, TValue* val,
                      TValue* default_v, int count) {
        ev->LookupOrCreate(key, val, default_v, count);
        return Status::OK();
      };
    } else {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key, TValue* val,
                      TValue* default_v, int count) {
        ev->LookupOrCreate(key, val, default_v);
        return Status::OK();
      };
    }
  }

  void Compute(OpKernelContext* ctx) override {
    /*
      step 1: unique and assign unique output and index
      step 2: doing parallel unique value gather
    */
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    // ScopedPerThreadMaxParallelism(num_lookups_ * 2);
    // uint64_t op_start_micros = Env::Default()->NowMicros();
    auto do_compute = [this, ctx, worker_threads](int i) {
      EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, i), &embedding_var));
      core::ScopedUnref unref_me(embedding_var);

      const Tensor& dense_values_tensor = ctx->input(num_lookups_ + i);
      auto dense_values = dense_values_tensor.flat<TKey>().data();
      int nnz = dense_values_tensor.NumElements();

      OP_REQUIRES(
          ctx,
          !embedding_var->IsMultiLevel() || (embedding_var->IsMultiLevel() &&
                                             embedding_var->CacheSize() >= nnz),
          errors::InvalidArgument("MultiLevel EV's Cache size ",
                                  embedding_var->CacheSize(),
                                  " should large than IDs in batch ", nnz));

      TensorShape unique_idx_tensor_shape =
          TensorShape(std::vector<int64>({static_cast<long long>(nnz)}));
      Tensor* unique_idx_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2 * num_lookups_ + i,
                                               unique_idx_tensor_shape,
                                               &unique_idx_tensor));
      auto unique_idx = unique_idx_tensor->flat<int>().data();

      // Stage 1
      /***********Doing Unique **************/
      google::dense_hash_map<TKey, int32> unique_map;
      unique_map.set_empty_key(std::numeric_limits<TKey>::max());
      unique_map.resize(2 * nnz);

      for (int64 k = 0, j = 0; k < nnz; ++k) {
        auto it = unique_map.emplace(dense_values[k], j);
        unique_idx[k] = it.first->second;

        if (it.second) {
          ++j;
        }
      }

      int unique_nnz = unique_map.size();
      Tensor* unique_tensor = nullptr;
      TensorShape unique_shape{static_cast<int64>(unique_nnz)};
      OP_REQUIRES_OK(ctx, ctx->allocate_output(num_lookups_ + i, unique_shape,
                                               &unique_tensor));
      auto* unique = unique_tensor->flat<TKey>().data();

      for (auto it : unique_map) {
        unique[it.second] = it.first;
      }

      TValue* default_v = nullptr;
      if (is_use_default_value_tensor_) {
        default_v =
            reinterpret_cast<TValue*>(ctx->input(num_lookups_ * 2 + 1).data());
      } else {
        default_v = embedding_var->GetDefaultValuePtr();
      }

      auto dense_values_tensor_shape = dense_values_tensor.shape();
      TensorShape emb_vectors_tensor_shape =
          TensorShape(dense_values_tensor_shape);
      emb_vectors_tensor_shape.AddDim(dimension_);
      Tensor* gather_embedding_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &gather_embedding_tensor));
      auto gather_embedding = gather_embedding_tensor->flat<TValue>().data();

      int slice_bytes = nnz * dimension_ * 1000;
      auto do_lookup = [this, ctx, embedding_var, unique, default_v, unique_idx,
                        gather_embedding](int64 start, int64 end) {
        for (int k = start; k < end; ++k) {
	  auto indices = unique_idx[k];
          TKey unique_id = unique[indices];
          TValue* default_v_ptr = get_default_v_fn_(
              default_v, unique_id, indices, embedding_var->GetDefaultValueDim(),
              embedding_var->ValueLen());
          OP_REQUIRES_OK(ctx, lookup_fn_(embedding_var, unique_id,
                                         gather_embedding + k * dimension_,
                                         default_v_ptr, 1/*count*/));
        }
      };
      Shard(worker_threads->num_threads, worker_threads->workers, nnz,
            slice_bytes, do_lookup);  // Parallel on batch

      if (embedding_var->IsMultiLevel()) {
        embedding::BatchCache<TKey>* cache = embedding_var->Cache();
        embedding_var->storage_manager()->Schedule(
            [embedding_var, dense_values_tensor] {
              embedding::BatchCache<TKey>* cache = embedding_var->Cache();
              cache->add_to_rank(dense_values_tensor);
            });
      }
      // LOG(INFO) << "idx " << i << " embedding time stats: "
      //           << " unique_secs: " <<
      //           strings::HumanReadableElapsedTime(unique_secs)
      //           << " lookup_secs: " <<
      //           strings::HumanReadableElapsedTime(lookup_secs)
      //           << " pooling_secs: " <<
      //           strings::HumanReadableElapsedTime(pooling_secs);
    };
    ParallelFor(do_compute, num_lookups_, thread_pool_.get());
  }

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::function<TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
  std::function<Status(EmbeddingVar<TKey, TValue>* ev, TKey key, TValue* val,
                       /*TValue* sp_weight,*/ TValue* default_v, int count)>
      lookup_fn_;
  // float max_norm_;
  int num_lookups_;
  int dimension_;
  bool is_use_default_value_tensor_;
  bool is_inference_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type) \
  REGISTER_KERNEL_BUILDER(                         \
      Name("GroupEmbeddingVarLookupDense")         \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<key_type>("Tkeys")       \
          .TypeConstraint<value_type>("dtype"),    \
      GroupEmbeddingVariableLookupDenseCpuOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

}  // namespace tensorflow

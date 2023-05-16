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

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/group_embedding/group_embedding_lookup_sparse_forward_base_ops.h"
#include "tensorflow/core/util/work_sharder.h"
namespace tensorflow {

#define USING_BASE_CLASS_MEMBER                                            \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_num_lookup;                  \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_dimension;                   \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_is_use_default_value_tensor; \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_is_sequence;                 \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_get_default_v_fn;            \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_lookup_fn;

template <typename TKey, typename TValue>
class GroupEmbeddingVariableLookupCpuOp
    : public GroupLookupBaseCpuOp<TKey, TValue> {
  USING_BASE_CLASS_MEMBER

 public:
  explicit GroupEmbeddingVariableLookupCpuOp(OpKernelConstruction *c)
      : GroupLookupBaseCpuOp<TKey, TValue>(c) {
    bool is_inference;
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference));

    if (!is_inference) {
      m_lookup_fn = [](EmbeddingVar<TKey, TValue> *ev, TKey key, TValue *val,
                       TValue *default_v, int count) {
        ev->LookupOrCreate(key, val, default_v, count);
        return Status::OK();
      };
    } else {
      m_lookup_fn = [](EmbeddingVar<TKey, TValue> *ev, TKey key, TValue *val,
                       TValue *default_v, int count) {
        ev->LookupOrCreate(key, val, default_v);
        return Status::OK();
      };
    }

    OP_REQUIRES_OK(c, c->GetAttr("is_use_default_value_tensor",
                                 &m_is_use_default_value_tensor));

    if (m_is_use_default_value_tensor) {
      m_get_default_v_fn = [](TValue *default_v, TKey id, int64 index,
                              int64 total_dim,
                              int64 len) { return default_v + len * index; };
    } else {
      m_get_default_v_fn = [](TValue *default_v, TKey id, int64 index,
                              int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim);
      };
    }
  }

  void Compute(OpKernelContext *ctx) override {
    /*
      step 1: unique and assign unique output and index
      step 2: doing unique value gather
      step 3: assign unique embedding to batch result and pooling
    */
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();

    for (int i = 0; i < m_num_lookup; ++i) {
      EmbeddingVar<TKey, TValue> *embedding_var = nullptr;
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, i), &embedding_var));
      core::ScopedUnref unref_me(embedding_var);

      const Tensor &sp_values_tensor = ctx->input(m_num_lookup + i);
      const Tensor &sp_indices_tensor = ctx->input(m_num_lookup * 2 + i);
      auto sp_indices = sp_indices_tensor.flat<int64>().data();
      int nnz = sp_values_tensor.NumElements();
      const Tensor &dense_shape_tensor = ctx->input(m_num_lookup * 4 + i);
      auto dense_shape = dense_shape_tensor.flat<int64>().data();
      int64 batch_size = dense_shape[0];

      OP_REQUIRES(
          ctx,
          !embedding_var->IsMultiLevel() || (embedding_var->IsMultiLevel() &&
                                             embedding_var->CacheSize() >= nnz),
          errors::InvalidArgument("MultiLevel EV's Cache size ",
                                  embedding_var->CacheSize(),
                                  " should large than IDs in batch ", nnz));

      // Stage 1
      Tensor unique_idx_tensor;
      Tensor unique_tensor;
      Tensor unique_counter;

      UniqueWithoutAxis<TKey, int32>(ctx, sp_values_tensor, &unique_idx_tensor,
                                     &unique_tensor, &unique_counter, 0,
                                     this->partition_size_, this->serial_,
                                     this->unique_ratio_hint_, this->map_flag_);

      ctx->set_output(m_num_lookup + i, unique_tensor);
      ctx->set_output(2 * m_num_lookup + i, unique_idx_tensor);

      auto *unique = unique_tensor.flat<TKey>().data();
      auto *unique_idx = unique_idx_tensor.flat<int>().data();

      int unique_nnz = unique_tensor.shape().dim_size(0);
      TensorShape unique_shape{static_cast<int64>(unique_nnz)};

      TensorShape batch_nums_tensor_shape =
          TensorShape(std::vector<int64>({batch_size}));
      Tensor *batch_nums_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(3 * m_num_lookup + i,
                                               batch_nums_tensor_shape,
                                               &batch_nums_tensor));
      auto batch_nums = batch_nums_tensor->flat<int>().data();
      memset(batch_nums, 0, batch_size * sizeof(int));
      for (int k = 0; k < nnz; ++k) {
        int batch_id = sp_indices[k * dense_shape_tensor.NumElements()];
        batch_nums[batch_id] += 1;
      }
      for (int k = 1; k < batch_size; ++k) {
        batch_nums[k] += batch_nums[k - 1];
      }

      TValue *default_v = nullptr;
      if (m_is_use_default_value_tensor) {
        default_v =
            reinterpret_cast<TValue *>(ctx->input(m_num_lookup * 4 + 1).data());
      } else {
        default_v = embedding_var->GetDefaultValuePtr();
      }

      // Stage 2
      Tensor unique_embedding;
      unique_shape.AppendShape({static_cast<int64>(m_dimension)});
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<TValue>::v(), unique_shape,
                                  &unique_embedding, attr));
      auto unique_embedding_data = unique_embedding.flat<TValue>().data();

      int slice_bytes = unique_nnz * m_dimension * 1000;
      auto do_lookup = [this, ctx, embedding_var, unique, default_v,
                        unique_embedding_data](int64 start, int64 end) {
        for (int k = start; k < end; ++k) {
          TValue *default_v_ptr = m_get_default_v_fn(
              default_v, unique[k], k, embedding_var->GetDefaultValueDim(),
              embedding_var->ValueLen());
          OP_REQUIRES_OK(ctx,
                         m_lookup_fn(embedding_var, unique[k],
                                     unique_embedding_data + k * m_dimension,
                                     default_v_ptr, 1 /*count*/));
        }
      };
      Shard(worker_threads->num_threads, worker_threads->workers, unique_nnz,
            slice_bytes /*cost*/, do_lookup);

      if (embedding_var->IsMultiLevel()) {
        embedding::BatchCache<TKey> *cache = embedding_var->Cache();
        embedding_var->storage()->Schedule(
            [embedding_var, sp_values_tensor] {
              embedding::BatchCache<TKey> *cache = embedding_var->Cache();
              cache->add_to_rank(sp_values_tensor);
            });
      }

      std::vector<TValue> default_weights(nnz, 1.0);
      TValue *sp_weights = default_weights.data();
      if (!this->m_ignore_weights) {
        const Tensor &sp_weights_tensor =
            ctx->input(this->m_num_lookup * 3 + i);
        sp_weights =
            const_cast<TValue *>(sp_weights_tensor.flat<TValue>().data());
      }

      // Stage 3
      TensorShape emb_vectors_tensor_shape;
      // Special case for sequence categorical column output
      if (m_is_sequence) {
        emb_vectors_tensor_shape = TensorShape(
            std::vector<int64>({batch_size, dense_shape[1], m_dimension}));
      } else {
        emb_vectors_tensor_shape =
            TensorShape(std::vector<int64>({batch_size, m_dimension}));
      }
      Tensor *gather_embedding_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &gather_embedding_tensor));
      auto gather_embedding = gather_embedding_tensor->flat<TValue>().data();

      slice_bytes = nnz / batch_size * m_dimension * 1000;
      // todo: clean these redundant code
      if (this->m_combiner == "mean") {
        auto embedding_var_mean_combiner = [this, &gather_embedding, batch_nums,
                                            unique_idx, unique,
                                            unique_embedding_data, sp_weights](
                                               int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
            int tmp_length = (m_dimension + 15) / 16;
            __m512 tmp_embedding[tmp_length];
            for (int i = 0; i < tmp_length; ++i) {
              tmp_embedding[i] = _mm512_set1_ps(0.0f);
            }
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            __m512 _bs = _mm512_set1_ps(batch_num);
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              float *u_embedding =
                  unique_embedding_data + unique_indice * m_dimension;
              __m512 _weights =
                  _mm512_set1_ps(*(sp_weights + batch_offset + j));
              _weights = _mm512_div_ps(_weights, _bs);
              for (int d = 0; d < m_dimension; d += 16) {
                int index = d / 16;
                int remain = m_dimension - d;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 _item = _mm512_maskz_loadu_ps(mask, u_embedding + d);
                tmp_embedding[index] = _mm512_mask3_fmadd_ps(
                    _item, _weights, tmp_embedding[index], mask);
              }
            }

            for (int d = 0; d < m_dimension; d += 16) {
              int index = d / 16;
              int remain = m_dimension - d;
              __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
              _mm512_mask_storeu_ps(gather_embedding + i * m_dimension + d,
                                    mask, tmp_embedding[index]);
            }
#else
            std::vector<TValue> tmp_embedding(m_dimension, 0.0f);
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              float *u_embedding =
                  unique_embedding_data + unique_indice * m_dimension;
              TValue sp_weight = sp_weights[batch_offset + j] / batch_num;
              for (int d = 0; d < m_dimension; ++d) {
                tmp_embedding[d] =
                    std::fma(*(u_embedding + d), sp_weight, tmp_embedding[d]);
              }
            }
            memcpy(gather_embedding + i * m_dimension, tmp_embedding.data(),
                   sizeof(float) * m_dimension);
#endif
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/, embedding_var_mean_combiner);
      } else if (this->m_combiner == "sum") {
        auto embedding_var_sum_combiner = [this, &gather_embedding, batch_nums,
                                           unique_idx, unique,
                                           unique_embedding_data,
                                           sp_weights](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
            int tmp_length = (m_dimension + 15) / 16;
            __m512 tmp_embedding[tmp_length];
            for (int i = 0; i < tmp_length; ++i) {
              tmp_embedding[i] = _mm512_set1_ps(0.0f);
            }
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              float *u_embedding =
                  unique_embedding_data + unique_indice * m_dimension;
              __m512 _weights =
                  _mm512_set1_ps(*(sp_weights + batch_offset + j));
              for (int d = 0; d < m_dimension; d += 16) {
                int index = d / 16;
                int remain = m_dimension - d;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 _item = _mm512_maskz_loadu_ps(mask, u_embedding + d);
                tmp_embedding[index] = _mm512_mask3_fmadd_ps(
                    _item, _weights, tmp_embedding[index], mask);
              }
            }
            for (int d = 0; d < m_dimension; d += 16) {
              int index = d / 16;
              int remain = m_dimension - d;
              __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
              _mm512_mask_storeu_ps(gather_embedding + i * m_dimension + d,
                                    mask, tmp_embedding[index]);
            }
#else
            std::vector<TValue> tmp_embedding(m_dimension, 0.0f);
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              float *u_embedding =
                  unique_embedding_data + unique_indice * m_dimension;
              for (int d = 0; d < m_dimension; ++d) {
                tmp_embedding[d] =
                    std::fma(u_embedding[d], sp_weights[batch_offset + j],
                             tmp_embedding[d]);
              }
            }
            memcpy(gather_embedding + i * m_dimension, tmp_embedding.data(),
                   sizeof(float) * m_dimension);
#endif
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/, embedding_var_sum_combiner);
      } else {
        auto embedding_var_sqrtn_combiner = [this, &gather_embedding,
                                             batch_nums, unique_idx, unique,
                                             unique_embedding_data, sp_weights](
                                                int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
            int tmp_length = (m_dimension + 15) / 16;
            __m512 tmp_embedding[tmp_length];
            for (int i = 0; i < tmp_length; ++i) {
              tmp_embedding[i] = _mm512_set1_ps(0.0f);
            }
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            __m512 _bs = _mm512_set1_ps(sqrtf(batch_num));
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              float *u_embedding =
                  unique_embedding_data + unique_indice * m_dimension;
              __m512 _weights =
                  _mm512_set1_ps(*(sp_weights + batch_offset + j));
              _weights = _mm512_div_ps(_weights, _bs);
              for (int d = 0; d < m_dimension; d += 16) {
                int index = d / 16;
                int remain = m_dimension - d;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 _item = _mm512_maskz_loadu_ps(mask, u_embedding + d);
                tmp_embedding[index] = _mm512_mask3_fmadd_ps(
                    _item, _weights, tmp_embedding[index], mask);
              }
            }

            for (int d = 0; d < m_dimension; d += 16) {
              int index = d / 16;
              int remain = m_dimension - d;
              __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
              _mm512_mask_storeu_ps(gather_embedding + i * m_dimension + d,
                                    mask, tmp_embedding[index]);
            }
#else
            std::vector<TValue> tmp_embedding(m_dimension, 0.0f);
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              float *u_embedding =
                  unique_embedding_data + unique_indice * m_dimension;
              TValue sp_weight =
                  sp_weights[batch_offset + j] / sqrtf(batch_num);
              for (int d = 0; d < m_dimension; ++d) {
                tmp_embedding[d] =
                    std::fma(u_embedding[d], sp_weight, tmp_embedding[d]);
              }
            }
            memcpy(gather_embedding + i * m_dimension, tmp_embedding.data(),
                   sizeof(float) * m_dimension);
#endif
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/, embedding_var_sqrtn_combiner);
      }
    }
  }
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
class GroupVariableLookupCpuOp : public GroupLookupBaseCpuOp<TKey, TValue> {
  USING_BASE_CLASS_MEMBER
 public:
  explicit GroupVariableLookupCpuOp(OpKernelConstruction *c)
      : GroupLookupBaseCpuOp<TKey, TValue>(c) {}

  void Compute(OpKernelContext *ctx) override {
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    for (int i = 0; i < m_num_lookup; ++i) {
      const Tensor &emb_variable_tensor = ctx->input(i);
      const Tensor &sp_values_tensor = ctx->input(m_num_lookup + i);
      int nnz = sp_values_tensor.NumElements();
      auto embedding_variable = emb_variable_tensor.flat<TValue>().data();

      const Tensor &sp_indices_tensor = ctx->input(m_num_lookup * 2 + i);
      auto sp_indices = sp_indices_tensor.flat<int64>().data();

      const Tensor &dense_shape_tensor = ctx->input(m_num_lookup * 4 + i);
      auto dense_shape = dense_shape_tensor.flat<int64>().data();
      int64 batch_size = dense_shape[0];

      TensorShape batch_nums_tensor_shape =
          TensorShape(std::vector<int64>({batch_size}));
      Tensor *batch_nums_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(3 * m_num_lookup + i,
                                               batch_nums_tensor_shape,
                                               &batch_nums_tensor));
      auto batch_nums = batch_nums_tensor->flat<int>().data();
      memset(batch_nums, 0, batch_size * sizeof(int));
      for (int k = 0; k < nnz; ++k) {
        int batch_id = sp_indices[k * dense_shape_tensor.NumElements()];
        batch_nums[batch_id] += 1;
      }
      for (int k = 1; k < batch_size; ++k) {
        batch_nums[k] += batch_nums[k - 1];
      }

      TensorShape emb_vectors_tensor_shape;
      // Special case for sequence categorical column output
      if (m_is_sequence) {
        emb_vectors_tensor_shape = TensorShape(
            std::vector<int64>({batch_size, dense_shape[1], m_dimension}));
      } else {
        emb_vectors_tensor_shape =
            TensorShape(std::vector<int64>({batch_size, m_dimension}));
      }

      Tensor *emb_vectors_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &emb_vectors_tensor));
      auto emb_vectors = emb_vectors_tensor->flat<TValue>().data();

      // Stage 1
      Tensor unique_idx_tensor;
      Tensor unique_tensor;
      Tensor unique_counter;

      UniqueWithoutAxis<TKey, int32>(ctx, sp_values_tensor, &unique_idx_tensor,
                                     &unique_tensor, &unique_counter, 0,
                                     this->partition_size_, this->serial_,
                                     this->unique_ratio_hint_, this->map_flag_);

      ctx->set_output(m_num_lookup + i, unique_tensor);
      ctx->set_output(2 * m_num_lookup + i, unique_idx_tensor);

      auto *unique = unique_tensor.flat<TKey>().data();
      auto *unique_idx = unique_idx_tensor.flat<int>().data();

      std::vector<TValue> default_weights(nnz, 1.0);
      TValue *sp_weights = default_weights.data();
      if (!this->m_ignore_weights) {
        const Tensor &sp_weights_tensor =
            ctx->input(this->m_num_lookup * 3 + i);
        sp_weights =
            const_cast<TValue *>(sp_weights_tensor.flat<TValue>().data());
      }

      int slice_bytes = nnz / batch_size * m_dimension * 1000;
      if (this->m_combiner == "mean") {
        auto do_var_mean = [this, &emb_vectors, batch_nums, unique_idx, unique,
                            sp_weights,
                            embedding_variable](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
            int tmp_length = (m_dimension + 15) / 16;
            __m512 tmp_embedding[tmp_length];
            for (int i = 0; i < tmp_length; ++i) {
              tmp_embedding[i] = _mm512_set1_ps(0.0f);
            }
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              int unique_id = unique[unique_indice];
              __m512 _weights =
                  _mm512_set1_ps(*(sp_weights + batch_offset + j));
              __m512 _bs = _mm512_set1_ps(batch_num);
              _weights = _mm512_div_ps(_weights, _bs);
              const float *embedding_ptr =
                  embedding_variable + unique_id * m_dimension;

              for (int d = 0; d < m_dimension; d += 16) {
                int index = d / 16;
                int remain = m_dimension - d;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 _item = _mm512_maskz_loadu_ps(mask, embedding_ptr + d);
                tmp_embedding[index] = _mm512_mask3_fmadd_ps(
                    _item, _weights, tmp_embedding[index], mask);
              }
            }

            for (int d = 0; d < m_dimension; d += 16) {
              int index = d / 16;
              int remain = m_dimension - d;
              __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
              _mm512_mask_storeu_ps(emb_vectors + i * m_dimension + d, mask,
                                    tmp_embedding[index]);
            }
#else
            std::vector<TValue> tmp_embedding(m_dimension, 0.0f);
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              int unique_id = unique[unique_indice];
              TValue sp_weight = sp_weights[batch_offset + j] / batch_num;
              for (int d = 0; d < m_dimension; ++d) {
                tmp_embedding[d] =
                    std::fma(embedding_variable[unique_id * m_dimension + d],
                             sp_weight, tmp_embedding[d]);
              }
            }
            memcpy(emb_vectors + i * m_dimension, tmp_embedding.data(),
                   sizeof(float) * m_dimension);
#endif
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/, do_var_mean);
      } else if (this->m_combiner == "sum") {
        auto do_var_sum = [this, &emb_vectors, batch_nums, unique_idx, unique,
                           sp_weights,
                           embedding_variable](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
            int tmp_length = (m_dimension + 15) / 16;
            __m512 tmp_embedding[tmp_length];
            for (int i = 0; i < tmp_length; ++i) {
              tmp_embedding[i] = _mm512_set1_ps(0.0f);
            }
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              int unique_id = unique[unique_indice];
              __m512 _weights =
                  _mm512_set1_ps(*(sp_weights + batch_offset + j));
              const float *embedding_ptr =
                  embedding_variable + unique_id * m_dimension;
              for (int d = 0; d < m_dimension; d += 16) {
                int index = d / 16;
                int remain = m_dimension - d;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 _item = _mm512_maskz_loadu_ps(mask, embedding_ptr + d);
                tmp_embedding[index] = _mm512_mask3_fmadd_ps(
                    _item, _weights, tmp_embedding[index], mask);
              }
            }
            for (int d = 0; d < m_dimension; d += 16) {
              int index = d / 16;
              int remain = m_dimension - d;
              __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
              _mm512_mask_storeu_ps(emb_vectors + i * m_dimension + d, mask,
                                    tmp_embedding[index]);
            }
#else
            std::vector<TValue> tmp_embedding(m_dimension, 0.0f);
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              int unique_id = unique[unique_indice];
              for (int d = 0; d < m_dimension; ++d) {
                tmp_embedding[d] =
                    std::fma(embedding_variable[unique_id * m_dimension + d],
                             sp_weights[batch_offset + j], tmp_embedding[d]);
              }
            }
            memcpy(emb_vectors + i * m_dimension, tmp_embedding.data(),
                   sizeof(float) * m_dimension);
#endif
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/, do_var_sum);
      } else {
        auto do_var_sqrtn = [this, &emb_vectors, batch_nums, unique_idx, unique,
                             sp_weights,
                             embedding_variable](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
            int tmp_length = (m_dimension + 15) / 16;
            __m512 tmp_embedding[tmp_length];
            for (int i = 0; i < tmp_length; ++i) {
              tmp_embedding[i] = _mm512_set1_ps(0.0f);
            }
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              int unique_id = unique[unique_indice];
              __m512 _weights =
                  _mm512_set1_ps(*(sp_weights + batch_offset + j));
              __m512 _bs = _mm512_set1_ps(sqrtf(batch_num));
              _weights = _mm512_div_ps(_weights, _bs);
              const float *embedding_ptr =
                  embedding_variable + unique_id * m_dimension;
              for (int d = 0; d < m_dimension; d += 16) {
                int index = d / 16;
                int remain = m_dimension - d;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                __m512 _item = _mm512_maskz_loadu_ps(mask, embedding_ptr + d);
                tmp_embedding[index] = _mm512_mask3_fmadd_ps(
                    _item, _weights, tmp_embedding[index], mask);
              }
            }

            for (int d = 0; d < m_dimension; d += 16) {
              int index = d / 16;
              int remain = m_dimension - d;
              __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
              _mm512_mask_storeu_ps(emb_vectors + i * m_dimension + d, mask,
                                    tmp_embedding[index]);
            }
#else
            std::vector<TValue> tmp_embedding(m_dimension, 0.0f);
            int batch_offset = i == 0 ? 0 : batch_nums[i - 1];
            int batch_num = batch_nums[i] - batch_offset;
            for (int j = 0; j < batch_num; ++j) {
              int unique_indice = unique_idx[batch_offset + j];
              int unique_id = unique[unique_indice];
              TValue sp_weight =
                  sp_weights[batch_offset + j] / sqrtf(batch_num);
              for (int d = 0; d < m_dimension; ++d) {
                tmp_embedding[d] =
                    std::fma(embedding_variable[unique_id * m_dimension + d],
                             sp_weight, tmp_embedding[d]);
              }
            }
            memcpy(emb_vectors + i * m_dimension, tmp_embedding.data(),
                   sizeof(float) * m_dimension);
#endif
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
              slice_bytes /*cost*/, do_var_sqrtn);
      }
    }
  }
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

#undef USING_BASE_CLASS_MEMBER

}  // namespace tensorflow

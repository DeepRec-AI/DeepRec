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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda_runtime.h>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"
#include "tensorflow/core/kernels/group_embedding/group_embedding_lookup_sparse_forward_base_ops.cu.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/lib/core/spin_rw_lock.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

template <typename TFKey, typename TKey, typename TValue>
class GroupEmbeddingVarLookupOp
    : public GroupEmbeddingLookupForwardBaseOp<TKey, TValue> {
 public:
  explicit GroupEmbeddingVarLookupOp(OpKernelConstruction* c)
      : GroupEmbeddingLookupForwardBaseOp<TKey, TValue>(c) {
    OP_REQUIRES_OK(c, c->GetAttr("is_use_default_value_tensor",
                                 &is_use_default_value_tensor_));

    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TFKey id, int64 index,
                             int64 total_dim,
                             int64 len) { return default_v + len * index; };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TFKey id, int64 index,
                             int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim);
      };
    }
    bool is_inference;
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference));
    if (!is_inference) {
      lookup_fn_ = [](EmbeddingVar<TFKey, TValue>* ev, const TFKey* key,
                      TValue* val, TValue* default_v, int32 default_v_num,
                      bool is_use_default_value_tensor,
                      size_t n, const Eigen::GpuDevice& device) {
        ev->LookupOrCreate(key, val, default_v, default_v_num,
            is_use_default_value_tensor, n, device);
      };
    } else {
      lookup_fn_ = [](EmbeddingVar<TFKey, TValue>* ev, const TFKey* key,
                      TValue* val, TValue* default_v, int32 default_v_num,
                      bool is_use_default_value_tensor,
                      size_t n, const Eigen::GpuDevice& device) {
        ev->Lookup(key, val, default_v, default_v_num,
            is_use_default_value_tensor, n, device);
      };
    }

  }

  ~GroupEmbeddingVarLookupOp() { delete[] occupy_flag_; }

  void Compute(OpKernelContext* ctx) override {
    const auto& device = ctx->eigen_device<GPUDevice>();
    TValue* default_v = nullptr;
    int64 batch_size = -1;

    Allocator* gpu_allocator =
        ctx->device()->GetAllocator(AllocatorAttributes());
    GroupEmbeddingLookupForWard<TKey, TValue> lookuper(
        this->num_lookups_, this->dimension_, this->max_norm_, gpu_allocator);

    std::vector<Tensor> tensor_list;
    tensor_list.reserve(this->num_lookups_);

    for (int i = 0; i < this->num_lookups_; ++i) {
      EmbeddingVar<TFKey, TValue>* ev = nullptr;
      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, i), &ev));
      core::ScopedUnref unref_me(ev);
      int64 dimension = ev->ValueLen();

      const Tensor& sp_values_tensor = ctx->input(this->num_lookups_ + i);
      auto sp_values = sp_values_tensor.flat<TFKey>();
      int64 N = sp_values_tensor.NumElements();

      const Tensor& sp_indices_tensor = ctx->input(this->num_lookups_ * 2 + i);
      auto sp_indices = sp_indices_tensor.flat<int64>().data();
      int nnz = sp_indices_tensor.shape().dim_size(0);
      const Tensor& dense_shape_tensor = ctx->input(this->num_lookups_ * 4 + i);
      auto dense_shape = dense_shape_tensor.flat<int64>().data();
      int dense_shape_num = dense_shape_tensor.NumElements();
      batch_size = dense_shape[0];

      TValue* default_v = nullptr;
      if (is_use_default_value_tensor_) {
        default_v = (TValue*)ctx->input(5 * this->num_lookups_).data();
      } else {
        default_v = ev->GetDefaultValuePtr();
      }

      // DEBUG
      const TFKey* key_base = sp_values.data();
      Tensor out_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<TValue>::value,
                                             {N * dimension}, &out_tensor));
      TValue* out_base = out_tensor.flat<TValue>().data();

      if (ev->IsSingleHbm()) {
        if (is_use_default_value_tensor_) {
          Tensor default_values(ctx->input(5 * this->num_lookups_));
          auto default_value_num = default_values.NumElements() / dimension;
          auto default_values_matrix =
              default_values.shaped<TValue, 2>({default_value_num, dimension});
          TValue* default_v_base = &default_values_matrix(0, 0);
          lookup_fn_(ev, key_base, out_base, default_v_base,
                     default_value_num, is_use_default_value_tensor_, N,
                     device);
          
        } else {
          lookup_fn_(ev, key_base, out_base, ev->GetDefaultValuePtr(),
                     ev->GetDefaultValueDim(), true, N, device);
        }
      } else {
        auto out_flat =
            out_tensor.shaped<TValue, 2>({N, out_tensor.NumElements() / N});
        const int64 slice_elems = out_flat.dimension(1);
        const size_t slice_bytes = slice_elems * sizeof(TValue);
        TValue** memcpy_address = new TValue*[N];
        TFKey* indices_host = new TFKey[N];

        auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
        int64 num_threads = worker_threads->num_threads;
        if (occupy_flag_ == nullptr) {
          mutex_lock l(m_init_occupy_flag_);
          // double check
          if (occupy_flag_ == nullptr) {
            occupy_flag_ = new bool[num_threads];
            memset(occupy_flag_, 0, sizeof(bool) * num_threads);
          }
        }
        std::vector<std::list<int64>> init_cursor_list(num_threads + 1);
        std::vector<std::list<int64>> copyback_cursor_list(num_threads + 1);

        volatile bool is_cpu_indices_ready = false;
        // Copy ids from GPU to CPU for CPU Lookup.
        auto stream = ctx->op_device_context()->stream();
        auto event_mgr = ctx->device()->tensorflow_gpu_device_info()->event_mgr;

        se::DeviceMemoryBase gpu_src(const_cast<TFKey*>(key_base),
                                     N * sizeof(TFKey));
        stream->ThenMemcpy(indices_host, gpu_src, N * sizeof(TFKey));
        SyncWithEventMgr(stream, event_mgr);

        uint64 main_thread_id = Env::Default()->GetCurrentThreadId();
        auto do_work = [this, indices_host, out_base, slice_elems, ctx, ev,
                        memcpy_address, &init_cursor_list,
                        &copyback_cursor_list, main_thread_id,
                        num_threads](int64 start, int64 limit) {
          uint64 thread_id = Env::Default()->GetCurrentThreadId();
          int64 position;
          if (thread_id == main_thread_id) {
            position = num_threads;
          } else {
            position = -1;
            {
              spin_rd_lock l(mu_);
              auto iter = hash_map_.find(thread_id);
              if (iter != hash_map_.end()) {
                position = iter->second;
              }
            }

            if (position == -1) {
              // bind a new thread to a local cursor_list
              position = thread_id % num_threads;
              while (!__sync_bool_compare_and_swap(&(occupy_flag_[position]),
                                                   false, true)) {
                position = (position + 1) % num_threads;
              }
              {
                spin_wr_lock l(mu_);
                hash_map_.insert(std::pair<uint64, int64>(thread_id, position));
              }
            }
          }
          ev->LookupWithFreqBatch(indices_host, memcpy_address, start, limit,
                                  init_cursor_list[position],
                                  copyback_cursor_list[position]);
        };
        Shard(num_threads, worker_threads->workers, N, slice_bytes, do_work);
        for (int i = 1; i < num_threads + 1; i++) {
          if (init_cursor_list[i].size() > 0) {
            init_cursor_list[0].splice(init_cursor_list[0].end(),
                                       init_cursor_list[i]);
          }
          if (copyback_cursor_list[i].size() > 0) {
            copyback_cursor_list[0].splice(copyback_cursor_list[0].end(),
                                           copyback_cursor_list[i]);
          }
        }
        // Pointers in memcpy_address here will
        // be cast to ValuePtr<Tvalue>* in this funcation.
        ev->AllocateMemoryForNewFeatures(memcpy_address, init_cursor_list[0]);

        ev->SetDefaultValueOfNewFeatures(
            indices_host, N, init_cursor_list[0], memcpy_address, default_v,
            get_default_v_fn_, stream, event_mgr, ctx->eigen_gpu_device());

        ev->CopyEmbeddingsFromCPUToGPU(indices_host, copyback_cursor_list[0],
                                       memcpy_address, stream, event_mgr,
                                       ctx->eigen_gpu_device(), worker_threads);

        ev->CopyEmbeddingsToBuffer(out_base, N, slice_elems, memcpy_address,
                                   stream, event_mgr, ctx->eigen_gpu_device());
        delete[] memcpy_address;

        if (ev->IsMultiLevel()) {
          ev->storage()->Schedule([ev, indices_host, N]() {
            embedding::BatchCache<TFKey>* cache = ev->Cache();
            cache->update(indices_host, N);
            delete[] indices_host;
          });
        }
      }

      TensorShape emb_vectors_tensor_shape;
      // Special case for sequence categorical column output
      if (this->is_sequence_) {
        emb_vectors_tensor_shape = TensorShape(
            std::vector<int64>({batch_size, dense_shape[1], dimension}));
      } else {
        emb_vectors_tensor_shape =
            TensorShape(std::vector<int64>({batch_size, dimension}));
      }

      Tensor* op_output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &op_output_tensor));
      auto op_output = op_output_tensor->flat<TValue>().data();

      // allocate offset tensor
      TensorShape values_offset_tensor_shape =
          TensorShape(std::vector<int64>({batch_size}));

      // Fake Output
      Tensor* unique_keys_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {this->num_lookups_ + i}, this->num_lookups_ + i,
                              sp_values_tensor.shape(), &unique_keys_tensor));

      Tensor* unique_idx_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ * 2 + i,
                                               values_offset_tensor_shape,
                                               &unique_idx_tensor));

      Tensor* values_offset_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ * 3 + i,
                                               values_offset_tensor_shape,
                                               &values_offset_tensor));
      auto values_offset = values_offset_tensor->flat<int>().data();

      launch_cal_per_element_row_offset(
          batch_size, nnz, dense_shape_num, reinterpret_cast<const int64_t*>(sp_indices),
          values_offset, device.stream());

      TValue* sp_weights = nullptr;
      if (!this->ignore_weights_) {
        const Tensor& sp_weights_tensor =
            ctx->input(this->num_lookups_ * 3 + i);
        sp_weights =
            const_cast<TValue*>(sp_weights_tensor.flat<TValue>().data());
      }

      GroupEmbeddingForWardArgs<TKey, TValue> group_embedding_args(
          out_base, sp_weights, op_output,
          const_cast<TKey*>(reinterpret_cast<const TKey*>(key_base)),
          values_offset, nnz);

      lookuper.set(group_embedding_args);
      tensor_list.emplace_back(out_tensor);
    }

    if (this->combiner_ == "sum") {
      this->template compute<true, Sum>(lookuper, batch_size, device.stream());
    } else if (this->combiner_ == "mean") {
      this->template compute<true, Mean>(lookuper, batch_size, device.stream());
    } else {
      this->template compute<true, Sqrtn>(lookuper, batch_size,
                                          device.stream());
    }
  }

 private:
  std::map<uint64, int64> hash_map_;
  std::function<TValue*(TValue*, TFKey, int64, int64, int64)> get_default_v_fn_;
  std::function<void(EmbeddingVar<TFKey, TValue>* ev, const TFKey* key,
                      TValue* val, TValue* default_v, int32 default_v_num,
                      bool is_use_default_value_tensor,
                      size_t n, const Eigen::GpuDevice& device)> lookup_fn_;
  mutable easy_spinrwlock_t mu_ = EASY_SPINRWLOCK_INITIALIZER;
  bool* occupy_flag_{nullptr};
  mutex m_init_occupy_flag_;
  bool is_use_default_value_tensor_;
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("GroupEmbeddingVarLookup")                                \
          .Device(DEVICE_GPU)                                        \
          .HostMemory("dense_shape")                                 \ 
          .TypeConstraint<key_type_tf>("Tkeys")                      \
          .TypeConstraint<dtype>("dtype"),                           \
      GroupEmbeddingVarLookupOp<key_type_tf, key_type, dtype>)

REGISTER_GPU_KERNELS(int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, float, float);
#undef REGISTER_GPU_KERNELS

template <typename TFKey, typename TKey, typename TValue>
class GroupVariableLookupOp
    : public GroupEmbeddingLookupForwardBaseOp<TKey, TValue> {
 public:
  explicit GroupVariableLookupOp(OpKernelConstruction* c)
      : GroupEmbeddingLookupForwardBaseOp<TKey, TValue>(c) {}

  void Compute(OpKernelContext* ctx) override {
    const cudaStream_t stream = ctx->eigen_device<GPUDevice>().stream();
    Allocator* gpu_allocator =
        ctx->device()->GetAllocator(AllocatorAttributes());
    GroupEmbeddingLookupForWard<TKey, TValue> lookuper(
        this->num_lookups_, this->dimension_, this->max_norm_, gpu_allocator);
    int64 batch_size = -1;

    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor& emb_variable_tensor = ctx->input(i);
      const Tensor& sp_values_tensor = ctx->input(this->num_lookups_ + i);
      int64 emb_vec_size = emb_variable_tensor.shape().dim_size(1);

      const Tensor& sp_indices_tensor = ctx->input(this->num_lookups_ * 2 + i);
      auto sp_indices = sp_indices_tensor.flat<int64>().data();
      int nnz = sp_indices_tensor.shape().dim_size(0);
      const Tensor& dense_shape_tensor = ctx->input(this->num_lookups_ * 4 + i);
      auto dense_shape = dense_shape_tensor.flat<int64>().data();
      int dense_shape_num = dense_shape_tensor.NumElements();
      batch_size = dense_shape[0];

      TensorShape emb_vectors_tensor_shape;
      // Special case for sequence categorical column output
      if (this->is_sequence_) {
        emb_vectors_tensor_shape = TensorShape(
            std::vector<int64>({batch_size, dense_shape[1], emb_vec_size}));
      } else {
        emb_vectors_tensor_shape =
            TensorShape(std::vector<int64>({batch_size, emb_vec_size}));
      }
      Tensor* emb_vectors_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &emb_vectors_tensor));
      auto emb_vectors = emb_vectors_tensor->flat<TValue>().data();

      // allocate offset tensor
      TensorShape values_offset_tensor_shape =
          TensorShape(std::vector<int64>({batch_size}));
      // Fake Output
      Tensor* unique_keys_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {this->num_lookups_ + i}, this->num_lookups_ + i,
                              sp_values_tensor.shape(), &unique_keys_tensor));

      Tensor* unique_idx_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ * 2 + i,
                                               values_offset_tensor_shape,
                                               &unique_idx_tensor));
      Tensor* values_offset_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ * 3 + i,
                                               values_offset_tensor_shape,
                                               &values_offset_tensor));
      auto values_offset = values_offset_tensor->flat<int>().data();
      launch_cal_per_element_row_offset(
          batch_size, nnz, dense_shape_num, reinterpret_cast<const int64_t*>(sp_indices),
          values_offset, stream);

      TValue* sp_weights = nullptr;
      if (!this->ignore_weights_) {
        const Tensor& sp_weights_tensor =
            ctx->input(this->num_lookups_ * 3 + i);
        sp_weights =
            const_cast<TValue*>(sp_weights_tensor.flat<TValue>().data());
      }
      GroupEmbeddingForWardArgs<TKey, TValue> group_embedding_args(
          const_cast<TValue*>(emb_variable_tensor.flat<TValue>().data()),
          sp_weights, emb_vectors,
          const_cast<TKey*>(reinterpret_cast<const TKey*>(
              sp_values_tensor.flat<TFKey>().data())),
          values_offset, nnz);
      lookuper.set(group_embedding_args);
    }

    if (this->combiner_ == "sum") {
      this->template compute<false, Sum>(lookuper, batch_size, stream);
    } else if (this->combiner_ == "mean") {
      this->template compute<false, Mean>(lookuper, batch_size, stream);
    } else {
      this->template compute<false, Sqrtn>(lookuper, batch_size, stream);
    }
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(Name("GroupVariableLookup")                \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("dense_shape")             \ 
                              .TypeConstraint<key_type_tf>("Tkeys")  \
                              .TypeConstraint<dtype>("dtype"),       \
                          GroupVariableLookupOp<key_type_tf, key_type, dtype>)

REGISTER_GPU_KERNELS(int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, float, float);
#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

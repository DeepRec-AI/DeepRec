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
#endif

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif //GOOGLE_CUDA

namespace tensorflow {

#if GOOGLE_CUDA
using se::DeviceMemoryBase;
using se::Stream;
#endif //GOOGLE_CUDA

namespace {
const char* kInferenceMode = "INFERENCE_MODE";
}

template <typename TKey, typename TValue>
class KvResourceLookupResourceOp : public OpKernel {
 public:
  explicit KvResourceLookupResourceOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {1}, &output));
    auto output_scalar = output->scalar<int64>();
    output_scalar() = (int64)ev;
  }
};

#define REGISTER_KV_LOOKUP_RESOURCE(dev, ktype, vtype)                 \
  REGISTER_KERNEL_BUILDER(Name("KvResourceLookupResource")             \
                          .Device(DEVICE_##dev)                        \
                          .HostMemory("output")                        \
                          .TypeConstraint<ktype>("Tkeys")              \
                          .TypeConstraint<vtype>("dtype"),             \
                          KvResourceLookupResourceOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)                                \
  REGISTER_KV_LOOKUP_RESOURCE(dev, int32, type)                        \
  REGISTER_KV_LOOKUP_RESOURCE(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KV_LOOKUP_RESOURCE

template <typename TKey, typename TValue>
class KvResourceLookupIDOp : public OpKernel {
 public:
  explicit KvResourceLookupIDOp(OpKernelConstruction* c) : OpKernel(c) {
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = c->input(1);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    if (N > 0) {
      auto out_flat = out->flat<int64>();
      int64* out_base = &out_flat(0);

      auto indices_flat = indices.flat<TKey>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      auto do_work = [this, indices_flat,
           out_base, ev] (int64 start, int64 limit) {
        for (int64 i = start; i < limit; ++i) {
          ValuePtr<TValue>* value_ptr;
          bool is_filter = false;
          ev->LookupOrCreateKey(indices_flat(i), &value_ptr, &is_filter, false);
          *(out_base + i) = (int64)value_ptr;
        }
      };

      auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads->num_threads, worker_threads->workers, indices_size,
          100, do_work);

      if (ev->IsMultiLevel()) {
        ev->storage_manager()->Schedule([ev, indices]() {
          embedding::BatchCache<TKey>* cache = ev->Cache();
          if (cache) {
            cache->add_to_rank(indices);
          }
        });
      }
    }
  }
};

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceLookupID")         \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceLookupIDOp<ktype, vtype>)
#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceCollectEmbeddingOp : public OpKernel {
 public:
  explicit KvResourceCollectEmbeddingOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * index;
      };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim) ;
      };
    }
    if (c->num_inputs() == 5) {
      get_count_fn_ = [](const int32* count, int64 index) {
        return count[index];
      };
    } else {
      get_count_fn_ = [](const int32* count, int64 index) {
        return 1;
      };
    }
    lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key,
                    TValue* val, TValue* default_v, int count) {
      if (key) {
        TValue* mem_val = ev->LookupOrCreateEmb((ValuePtr<TValue>*)key, default_v);
        memcpy(val, mem_val, sizeof(TValue) * ev->ValueLen());
      } else {
        memcpy(val, default_v, sizeof(TValue) * ev->ValueLen());
      }
      return Status::OK();
    };
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = c->input(1);
    const Tensor& pointer = c->input(2);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({ev->ValueLen()});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    int32* counts = nullptr;
    if (c->num_inputs() == 5)
      counts = (int32*)c->input(4).data();

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      auto pointer_flat = pointer.flat<int64>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      TValue* default_v = nullptr;
      if (is_use_default_value_tensor_) {
        default_v = (TValue*)c->input(3).data();
      } else {
        default_v = ev->GetDefaultValuePtr();
      }
      OP_REQUIRES(c, ev->ValueLen() == slice_elems,
          errors::InvalidArgument(
              "ev's value_len should same with output's dimension(1)",
              std::to_string(slice_elems), std::to_string(ev->ValueLen())));
      OP_REQUIRES(c, !ev->IsMultiLevel() ||
          (ev->IsMultiLevel() && ev->CacheSize() >= N),
          errors::InvalidArgument(
              "MultiLevel EV's Cache size ", ev->CacheSize(),
              " should large than IDs in batch ", N));
      const size_t slice_bytes = slice_elems * sizeof(TValue);
      auto do_work = [this, indices_flat, pointer_flat,
           out_base, slice_elems, c, default_v, ev, counts] (
               int64 start, int64 limit) {
        for (int64 i = start; i < limit; ++i) {
          TValue* default_v_ptr = get_default_v_fn_(
              default_v, indices_flat(i), i, ev->GetDefaultValueDim(),
              ev->ValueLen());
          int32 count = get_count_fn_(counts, i);
          OP_REQUIRES_OK(c, lookup_fn_(ev, pointer_flat(i),
              out_base + i * slice_elems, default_v_ptr, count));
        }
      };
      auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads->num_threads,
            worker_threads->workers, indices_size,
            slice_bytes, do_work);
    }
  }

  private:
    bool is_use_default_value_tensor_;
    std::function<
      TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
    std::function<int32(int32*, int64)> get_count_fn_;
    std::function<Status(EmbeddingVar<TKey, TValue>* ev,
      TKey key, TValue* val, TValue* default_v, int count)> lookup_fn_;
};

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceCollectEmbedding") \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("pointer")              \
                              .HostMemory("default_value")        \
                              .HostMemory("output")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceCollectEmbeddingOp<ktype, vtype>)

#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceGatherOp : public OpKernel {
 public:
  explicit KvResourceGatherOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("is_inference", &is_inference_));
    bool is_inference;
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference));
    is_inference_ |= is_inference;
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * index;
      };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim) ;
      };
    }
    if (c->num_inputs() == 4) {
      get_count_fn_ = [](const int32* count, int64 index) {
        return count[index];
      };
    } else {
      get_count_fn_ = [](const int32* count, int64 index) {
        return 1;
      };
    }
    if (!is_inference_) {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key,
                      TValue* val, TValue* default_v, int count) {
        ev->LookupOrCreate(key, val, default_v, count);
        return Status::OK();
      };
    } else {
      lookup_fn_ = [](EmbeddingVar<TKey, TValue>* ev, TKey key,
                      TValue* val, TValue* default_v, int count) {
        Status s = ev->Lookup(key, val, default_v);
        return s;
      };
    }
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = c->input(1);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({ev->ValueLen()});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    int32* counts = nullptr;
    if (c->num_inputs() == 4)
      counts = (int32*)c->input(3).data();

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      TValue* default_v = nullptr;
      if (is_use_default_value_tensor_) {
        default_v = (TValue*)c->input(2).data();
      } else {
        default_v = ev->GetDefaultValuePtr();
      }
      OP_REQUIRES(c, ev->ValueLen() == slice_elems,
          errors::InvalidArgument(
              "ev's value_len should same with output's dimension(1)",
              std::to_string(slice_elems), std::to_string(ev->ValueLen())));
      OP_REQUIRES(c, !ev->IsMultiLevel() ||
          (ev->IsMultiLevel() && ev->CacheSize() >= N),
          errors::InvalidArgument(
              "MultiLevel EV's Cache size ", ev->CacheSize(),
              " should large than IDs in batch ", N));
      const size_t slice_bytes = slice_elems * sizeof(TValue);
      auto do_work = [this, indices_flat,
           out_base, slice_elems, c, default_v, ev, counts] (
               int64 start, int64 limit) {
        for (int64 i = start; i < limit; ++i) {
          TValue* default_v_ptr = get_default_v_fn_(
              default_v, indices_flat(i), i, ev->GetDefaultValueDim(),
              ev->ValueLen());
          int32 count = get_count_fn_(counts, i);
          OP_REQUIRES_OK(c, lookup_fn_(ev, indices_flat(i),
              out_base + i * slice_elems, default_v_ptr, count));
        }
      };
      auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads->num_threads,
            worker_threads->workers, indices_size,
            slice_bytes, do_work);

      if (ev->IsMultiLevel()) {
        embedding::BatchCache<TKey>* cache = ev->Cache();
        ev->storage_manager()->Schedule([ev, indices]() {
          embedding::BatchCache<TKey>* cache = ev->Cache();
          cache->add_to_rank(indices);
        });
      }
    }
  }

  private:
    bool is_use_default_value_tensor_;
    bool is_inference_;
    std::function<
      TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
    std::function<int32(int32*, int64)> get_count_fn_;
    std::function<Status(EmbeddingVar<TKey, TValue>* ev,
      TKey key, TValue* val, TValue* default_v, int count)> lookup_fn_;
};

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGather")                \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("default_value")        \
                              .HostMemory("output")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOp<ktype, vtype>)

#define REGISTER_KERNELS_ALL_INDICES(type)                        \
  REGISTER_KERNELS(CPU, int32, type);                             \
  REGISTER_KERNELS(CPU, int64, type)

TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDICES)
#undef REGISTER_KERNELS_ALL_INDICES
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
template <typename Device, typename TKey, typename TValue>
class KvResourceGatherGPUOp : public OpKernel {
 public:
  explicit KvResourceGatherGPUOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c,
        c->GetAttr("is_use_default_value_tensor",
          &is_use_default_value_tensor_));
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * index;
      };
    } else {
      get_default_v_fn_ = [](TValue* default_v, TKey id, int64 index,
                            int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim) ;
      };
    }
    if (c->num_inputs() == 4) {
      get_count_fn_ = [](const int32* count, int64 index) {
        return count[index];
      };
    } else {
      get_count_fn_ = [](const int32* count, int64 index) {
        return 1;
      };
    }
    hash_map_.max_load_factor(0.8);
    hash_map_.set_empty_key_and_value(-1, -1);
    hash_map_.set_counternum(16);
    hash_map_.set_deleted_key(-2);
  }

  ~KvResourceGatherGPUOp() {
    delete[] occupy_flag_;
  }

  void Compute(OpKernelContext* c) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = c->input(1);
    const int64 N = indices.NumElements();

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({ev->ValueLen()});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));

    int32* counts = nullptr;
    if (c->num_inputs() == 4)
      counts = (int32*)c->input(3).data();

    int64 num_threads = c->device()
                         ->tensorflow_cpu_worker_threads()
                         ->num_threads;
    if (occupy_flag_ == nullptr) {
      mutex_lock l(m_init_occupy_flag_);
      //double check
      if (occupy_flag_ == nullptr) {
        occupy_flag_ = new bool[num_threads];
        memset(occupy_flag_, 0, sizeof(bool) * num_threads);
      }
    }

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      TValue* default_v = nullptr;
      if (is_use_default_value_tensor_) {
        default_v = (TValue*)c->input(2).data();
      } else {
        default_v = ev->GetDefaultValuePtr();
      }
      OP_REQUIRES(c, ev->ValueLen() == slice_elems,
          errors::InvalidArgument(
              "ev's value_len should same with output's dimension(1)",
              std::to_string(slice_elems), std::to_string(ev->ValueLen())));
      OP_REQUIRES(c, !ev->IsMultiLevel() ||
          (ev->IsMultiLevel() && ev->CacheSize() >= N),
          errors::InvalidArgument(
              "MultiLevel EV's Cache size ", ev->CacheSize(),
              " should large than IDs in batch ", N));
      const size_t slice_bytes = slice_elems * sizeof(TValue);
      if (ev->IsSingleHbm()) {
        const TKey* key_base = &indices_flat(0);
        const Device& device = c->eigen_device<Device>();
        if (is_use_default_value_tensor_) {
          Tensor default_values(c->input(2));
          auto default_value_num = default_values.NumElements() / ev->ValueLen();
          auto default_values_matrix = default_values.shaped<TValue, 2>(
              {default_value_num, ev->ValueLen()});
          TValue* default_v_base = &default_values_matrix(0, 0);
          ev->LookupOrCreate(key_base, out_base, default_v_base,
              default_value_num, is_use_default_value_tensor_,
              indices_size, device);
        } else {
          ev->LookupOrCreate(key_base, out_base, ev->GetDefaultValuePtr(),
              ev->GetDefaultValueDim(), is_use_default_value_tensor_,
              indices_size, device);
        }
      } else {
        TValue** memcpy_address = new TValue*[indices_size];
        TKey* indices_host = new TKey[N];
        volatile bool is_cpu_indices_ready = false;
        //Copy ids from GPU to CPU for CPU Lookup.
        auto stream = c->op_device_context()->stream();
        se::DeviceMemoryBase gpu_src(
            const_cast<TKey*>(&indices_flat(0)), N * sizeof(TKey));
        stream->ThenMemcpy(indices_host, gpu_src, N * sizeof(TKey));
        c->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
            stream, [&is_cpu_indices_ready]() {is_cpu_indices_ready = true;});
        while(!is_cpu_indices_ready) {}
        auto worker_threads = c->device()->tensorflow_cpu_worker_threads();
        std::vector<std::list<int64>> init_cursor_list(
                                          worker_threads->num_threads + 1);
        std::vector<std::list<int64>> copyback_cursor_list(
                                          worker_threads->num_threads + 1);
        int64 main_thread_id = Env::Default()->GetCurrentThreadId();
        auto do_work = [this, indices_host,
            out_base, slice_elems, c, ev,
            memcpy_address, &init_cursor_list,
            &copyback_cursor_list, main_thread_id,
            num_threads] (int64 start, int64 limit) {
          int64 thread_id = Env::Default()->GetCurrentThreadId();
          int position;
          if (thread_id == main_thread_id) {
            position = num_threads;
          } else {
            auto iter = hash_map_.find_wait_free(thread_id);
            if (iter.first == -1) {
            // bind a new thread to a local cursor_list
              position = thread_id % num_threads;
              while (!__sync_bool_compare_and_swap(&(occupy_flag_[position]),
                                                   false, true)) {
                position = (position + 1) % num_threads;
            }
              hash_map_.insert_lockless(
                        std::move(std::pair<int64, int>(thread_id, position)));
            } else {
              position = iter.second;
            }
          }
          ev->LookupWithFreqBatch(indices_host, memcpy_address,
                                  start, limit, init_cursor_list[position],
                                  copyback_cursor_list[position]);
        };
        Shard(worker_threads->num_threads, worker_threads->workers, indices_size,
            slice_bytes, do_work);
        for (int i = 1; i < worker_threads->num_threads + 1; i++) {
          if (init_cursor_list[i].size()>0) {
            init_cursor_list[0].splice(init_cursor_list[0].end(),
                                       init_cursor_list[i]);
          }
          if (copyback_cursor_list[i].size()>0) {
            copyback_cursor_list[0].splice(copyback_cursor_list[0].end(),
                                           copyback_cursor_list[i]);
          }
        }
        //Pointers in memcpy_address here will 
        //be cast to ValuePtr<Tvalue>* in this funcation.
        ev->AllocateMemoryForNewFeatures(
            memcpy_address,
            init_cursor_list[0]);

        ev->SetDefaultValueOfNewFeatures(
            indices_host, indices_size,
            init_cursor_list[0], memcpy_address,
            default_v, get_default_v_fn_);

        ev->CopyEmbeddingsFromCPUToGPU(
            indices_host,
            copyback_cursor_list[0],
            memcpy_address);

        ev->CopyEmbeddingsToBuffer(
            out_base, indices_size,
            slice_elems, memcpy_address);
        delete []memcpy_address;

        if (ev->IsMultiLevel()) {
          ev->storage_manager()->Schedule([ev, indices_host, N]() {
            embedding::BatchCache<TKey>* cache = ev->Cache();
            cache->add_to_rank(indices_host, N);
            delete []indices_host;
          });
        }
      }
    }
  }

  private:
    bool is_use_default_value_tensor_;
    std::function<
      TValue*(TValue*, TKey, int64, int64, int64)> get_default_v_fn_;
    std::function<int32(int32*, int64)> get_count_fn_;
    typedef google::dense_hash_map_lockless<int64, int> LockLessHashMap;
    LockLessHashMap hash_map_;
    bool* occupy_flag_ = nullptr;
    mutex m_init_occupy_flag_;
};

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGather")                \
                              .Device(DEVICE_##dev)               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherGPUOp<GPUDevice, ktype, vtype>)

#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU);
#undef REGISTER_KERNELS_GPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

#define REGISTER_KERNELS(dev, ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("KvResourceGatherV1")              \
                              .Device(DEVICE_##dev)               \
                              .HostMemory("resource")             \
                              .HostMemory("indices")              \
                              .HostMemory("default_value")        \
                              .HostMemory("counts")               \
                              .HostMemory("output")               \
                              .TypeConstraint<vtype>("dtype")     \
                              .TypeConstraint<ktype>("Tkeys"),    \
                          KvResourceGatherOp<ktype, vtype>)

#define REGISTER_KERNELS_ALL(dev, type)                           \
  REGISTER_KERNELS(dev, int32, type);                             \
  REGISTER_KERNELS(dev, int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class EVGetFrequencyOp : public OpKernel {
 public:
  explicit EVGetFrequencyOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = ctx->input(1);
    auto indices_flat = indices.flat<TKey>();

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {indices.NumElements()}, &output));
    for (int i = 0; i < indices.NumElements(); ++i) {
      int64 f = ev->GetFreq(indices_flat(i));
      output->flat<int64>()(i) = f;
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVGetFrequency")                \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("Tvalues"),  \
                          EVGetFrequencyOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class EVGetVersionOp : public OpKernel {
 public:
  explicit EVGetVersionOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = ctx->input(1);
    auto indices_flat = indices.flat<TKey>();

    Tensor* output;
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, {indices.NumElements()}, &output));
    for (int i = 0; i < indices.NumElements(); ++i) {
      int64 v = ev->GetVersion(indices_flat(i));
      output->flat<int64>()(i) = v;
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVGetVersion")                  \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("Tvalues"),  \
                          EVGetVersionOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceLookupTierOp : public OpKernel {
 public:
  explicit KvResourceLookupTierOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    const Tensor& indices = ctx->input(1);
    auto indices_flat = indices.flat<TKey>();

    Tensor* output;
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output(0, {indices.NumElements()}, &output));
    for (int i = 0; i < indices.NumElements(); ++i) {
      int v = ev->storage_manager()->LookupTier(indices_flat(i));
      output->flat<int>()(i) = v;
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("KvResourceLookupTier")          \
                            .Device(DEVICE_CPU)                 \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          KvResourceLookupTierOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("KvResourceLookupTier")          \
                            .Device(DEVICE_GPU)                 \
                            .HostMemory("ids")                  \
                            .HostMemory("output")               \
                            .TypeConstraint<ktype>("Tkeys")     \
                            .TypeConstraint<vtype>("dtype"),    \
                          KvResourceLookupTierOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(type)                              \
  REGISTER_KERNELS(int32, type)                                 \
  REGISTER_KERNELS(int64, type)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_ALL)
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

}  // namespace tensorflow


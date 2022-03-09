/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <unordered_map>
#include "third_party/cuco_hash_table/cuco/dynamic_map.cuh"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/kv_variable_ops_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace cg = cooperative_groups;

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
class gpu_hash_map_tf_allocator {
  mutable std::unordered_map<void*, Tensor> allocated_tensor_map;
  void status_check(bool status) const {
    OP_REQUIRES(context_, status,
            errors::InvalidArgument("TF error: context->allocate_temp failed"));
  }

public:
  OpKernelContext *context_;
  using value_type = T;

  gpu_hash_map_tf_allocator(OpKernelContext *context) : context_(context) {}

  template <class U>
  gpu_hash_map_tf_allocator(gpu_hash_map_tf_allocator<U> const& a) noexcept : context_(a.context_) {}

  value_type* allocate(size_t size) const {
    Tensor buf;
    long long int buf_size = (long long int)size * sizeof(value_type);
    tensorflow::Status status = context_->allocate_temp(DT_UINT8, TensorShape{buf_size}, &buf);
    status_check(status == tensorflow::Status::OK());
    auto flat = buf.flat<uint8>();
    void *ptr = (void *)flat.data();
    allocated_tensor_map.emplace(ptr, buf);
    return (value_type*)ptr;
  }

  void deallocate(value_type* ptr, size_t) const {
    allocated_tensor_map.erase(ptr);
  }
};

template <typename T, typename U>
bool operator==(gpu_hash_map_tf_allocator<T> const&, gpu_hash_map_tf_allocator<U> const&) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(gpu_hash_map_tf_allocator<T> const& lhs, gpu_hash_map_tf_allocator<U> const& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename KeyType, typename ValueType, typename Allocator = gpu_hash_map_tf_allocator<uint8_t>>
class DynamicHashTable {
public:
  cuco::dynamic_map<KeyType, ValueType, cuda::thread_scope_device, Allocator> map_;

  DynamicHashTable(size_t initial_capacity, KeyType empty_key_sentinel, ValueType empty_value_sentinel, Allocator alloc)
      : map_(initial_capacity, empty_key_sentinel, empty_value_sentinel, alloc) {
  }
  ~DynamicHashTable() {}
};

template <typename K, typename V>
GPUHashTable<K, V>::GPUHashTable(K empty_key_sentinel, OpKernelContext *context, size_t initial_capacity) {
  hash_table = new DynamicHashTable<K, int32>(initial_capacity, empty_key_sentinel, -1, gpu_hash_map_tf_allocator<uint8_t>(context));
  cudaMallocManaged(&start_idx, sizeof(cuda::atomic<std::size_t, cuda::thread_scope_device>));
}

template <typename K, typename V>
GPUHashTable<K, V>::~GPUHashTable() {
  delete hash_table;
  cudaFree(start_idx);
}

template class GPUHashTable<int32, float>;
template class GPUHashTable<int32, double>;
template class GPUHashTable<int64, float>;
template class GPUHashTable<int64, double>;

namespace functor {
using atomicT = cuda::atomic<std::size_t, cuda::thread_scope_device>;

template <uint32_t block_size,
          uint32_t tile_size,
          typename Key,
          typename mutableViewT,
          typename ViewT,
          typename Hash     = cuco::detail::MurmurHash3_32<Key>,
          typename KeyEqual = thrust::equal_to<Key>>
__global__ void kv_lookup_and_insert_key_kernel(const Key* key_first,
                                                int32* value_first,
                                                int32 num_items,
                                                mutableViewT* submap_mutable_views,
                                                ViewT* submap_views,
                                                uint32_t num_submaps,
                                                atomicT* num_successes,
                                                atomicT* start_idx,
                                                int32 idx_offset,
                                                Hash hash = Hash{},
                                                KeyEqual key_equal = KeyEqual{}) {
  extern __shared__ size_t thread_num_successes[];
  if (threadIdx.x < num_submaps)
    thread_num_successes[threadIdx.x] = 0;
  __syncthreads();

  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();
  __shared__ int32 writeBuffer[block_size / tile_size];
  int32 tmp;

  while(key_idx < num_items) {
    auto key = *(key_first + key_idx);
    int32 found_value = empty_value_sentinel;

    for(auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      auto found = submap_view.find(tile, key, hash, key_equal);
      if (found != submap_view.end()) {
        found_value = found->second;
        break;
      }
    }
    if (found_value == empty_value_sentinel) {
      if (tile.thread_rank() == 0) {
        tmp = start_idx->fetch_add(1);
      }
      found_value = tile.shfl(tmp, 0);
      auto insert_pair = cuco::pair_type<Key, int32>{key, (int32)(found_value + idx_offset)};
      for(auto i = 0; i < num_submaps; ++i) {
        if (submap_mutable_views[i].insert(tile, insert_pair, hash, key_equal) &&
            tile.thread_rank() == 0) {
          atomicAdd(reinterpret_cast<unsigned long long *>(thread_num_successes) + i, 1ull);
          break;
        }
      }
    }

    // if (tile.thread_rank() == 0) { writeBuffer[threadIdx.x / tile_size] = found_value; }
    // __syncthreads();
    // if (tile.thread_rank() == 0) {
    //   *(value_first + key_idx) = writeBuffer[threadIdx.x / tile_size];
    // }
    if (tile.thread_rank() == 0) {
      *(value_first + key_idx) = found_value;
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
  __syncthreads();

  if (threadIdx.x < num_submaps)
    num_successes[threadIdx.x] += thread_num_successes[threadIdx.x];
}

template <typename Key, typename V>
struct KvLookupInsertKey<GPUDevice, Key, V> {
  void operator()(const Key* key_first,
                  int32* value_first,
                  int32 num_items,
                  GPUHashTable<Key, V>* hash_table,
                  atomicT* start_idx,
                  int32 idx_offset,
                  cudaStream_t stream) {
    using mutableViewT = typename cuco::dynamic_map<Key, int32, cuda::thread_scope_device, gpu_hash_map_tf_allocator<uint8_t>>::mutable_view_type;
    using ViewT = typename cuco::dynamic_map<Key, int32, cuda::thread_scope_device, gpu_hash_map_tf_allocator<uint8_t>>::view_type;
    auto& map = hash_table->hash_table->map_;
    map.reserve(map.get_size() + num_items);
    for (size_t submap_idx = 0; submap_idx < map.get_submaps().size(); submap_idx++) {
      map.get_num_successes()[submap_idx] = 0;
    }
    int device_id;
    CUCO_CUDA_TRY(cudaGetDevice(&device_id));
    CUCO_CUDA_TRY(cudaMemPrefetchAsync(map.get_num_successes(), sizeof(atomicT) * map.get_submaps().size(), device_id));

    auto const block_size = 128;
    auto const stride = 1;
    auto const tile_size = 4;
    auto const grid_size = (tile_size * num_items + stride * block_size - 1) / (stride * block_size);
    TF_CHECK_OK(GpuLaunchKernel(kv_lookup_and_insert_key_kernel<block_size, tile_size, Key, mutableViewT, ViewT, cuco::detail::MurmurHash3_32<Key>, thrust::equal_to<Key>>,
                                grid_size, block_size, sizeof(size_t) * map.get_submaps().size(), stream,
                                key_first, value_first, num_items,
                                map.get_submap_mutable_views().data().get(), map.get_submap_views().data().get(), map.get_submaps().size(),
                                map.get_num_successes(), start_idx, idx_offset, cuco::detail::MurmurHash3_32<Key>{}, thrust::equal_to<Key>{}));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    map.update_submap_sizes();
  }
};

template <typename Value>
__global__ void kv_lookup_or_create_emb_kernel(Value* val,
                                               Value* default_v,
                                               int64 dim,
                                               int32* item_idxs,
                                               int32 slot_idx,
                                               Value** d_banks,
                                               bool** d_flags,
                                               int32 bank_num,
                                               int32 slot_num,
                                               int32 default_v_num,
                                               int32 bank_size) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto slot_offset = bank_idx * slot_num + slot_idx;
  if (bank_idx < bank_num && offset_in_bank < bank_size) {
    if (d_flags[slot_offset][offset_in_bank] == false) {
      d_flags[slot_offset][offset_in_bank] = true;
      for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
        d_banks[slot_offset][offset_in_bank * dim + id] = default_v[(item_idx % default_v_num) * dim + id];
      }
    }
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      val[item_idx * dim + id]= d_banks[slot_offset][slot_idx * dim + id];
    }
  }
}

template <typename Value>
struct KvLookupCreateEmb<GPUDevice, Value> {
  void operator()(Value* val,
                  Value* default_v,
                  int64 dim,
                  int32* item_idxs,
                  int32 num_items,
                  int32 slot_idx,
                  int32 default_v_num,
                  Value** d_banks,
                  bool** d_flags,
                  int32 bank_num,
                  int32 slot_num,
                  int32 bank_size,
                  cudaStream_t stream) {
  auto const block_size = 256;
  auto const grid_size = num_items;
  TF_CHECK_OK(GpuLaunchKernel(kv_lookup_or_create_emb_kernel<Value>,
                              grid_size, block_size, 0, stream,
                              val, default_v, dim,
                              item_idxs, slot_idx,
                              d_banks, d_flags,
                              bank_num, slot_num, default_v_num, bank_size));
}
};
} // namespace functor

template struct functor::KvLookupInsertKey<GPUDevice, int32, float>;
template struct functor::KvLookupInsertKey<GPUDevice, int32, double>;
template struct functor::KvLookupInsertKey<GPUDevice, int64, float>;
template struct functor::KvLookupInsertKey<GPUDevice, int64, double>;

template struct functor::KvLookupCreateEmb<GPUDevice, float>;
template struct functor::KvLookupCreateEmb<GPUDevice, double>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

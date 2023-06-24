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

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuco/dynamic_map.cuh"
#include "cuco/static_map.cuh"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/embedding/gpu_hash_table.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace cg = cooperative_groups;

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace {
const size_t BLOCK_SIZE = 128;
const size_t STRIDE = 1;
const size_t TILE_SIZE = 4;
}
template <typename T>
class gpu_hash_map_tf_allocator {
 public:
  Allocator* alloc_;
  using value_type = T;

  gpu_hash_map_tf_allocator(Allocator* alloc) : alloc_(alloc) {}

  gpu_hash_map_tf_allocator(const gpu_hash_map_tf_allocator& a) noexcept
      : alloc_(a.alloc_) {}

  template <typename U>
  gpu_hash_map_tf_allocator(const gpu_hash_map_tf_allocator<U>& a) noexcept
      : alloc_(a.alloc_) {}

  gpu_hash_map_tf_allocator& operator=(
      const gpu_hash_map_tf_allocator& a) noexcept {
    return *this;
  }

  gpu_hash_map_tf_allocator& operator=(gpu_hash_map_tf_allocator&& a) {
    alloc_ = a.alloc_;
    return *this;
  }

  ~gpu_hash_map_tf_allocator() noexcept {}

  value_type* allocate(size_t size) const {
    void* ptr =
        alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
                            size * sizeof(value_type), AllocationAttributes());
    return (value_type*)ptr;
  }

  void deallocate(value_type* ptr, size_t) const { alloc_->DeallocateRaw(ptr); }
};

template <typename T, typename U>
bool operator==(gpu_hash_map_tf_allocator<T> const&,
                gpu_hash_map_tf_allocator<U> const&) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(gpu_hash_map_tf_allocator<T> const& lhs,
                gpu_hash_map_tf_allocator<U> const& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename KeyType, typename ValueType,
          typename CUCOAllocator = gpu_hash_map_tf_allocator<uint8_t>>
class DynamicHashTable {
 public:
  cuco::dynamic_map<KeyType, ValueType, cuda::thread_scope_device,
                    CUCOAllocator>
      map_;

  DynamicHashTable(size_t initial_capacity, KeyType empty_key_sentinel,
                   ValueType empty_value_sentinel, CUCOAllocator alloc)
      : map_(initial_capacity, empty_key_sentinel, empty_value_sentinel,
             alloc) {}
  ~DynamicHashTable() {}
};

template <typename K, typename V>
GPUHashTable<K, V>::GPUHashTable(K empty_key_sentinel, Allocator* alloc,
                                 size_t initial_capacity)
    : initial_bank_size(initial_capacity) {
  hash_table =
      new DynamicHashTable<K, int32>(initial_capacity, empty_key_sentinel, -1,
                                     gpu_hash_map_tf_allocator<uint8_t>(alloc));
  cudaMallocManaged(
      &start_idx, sizeof(cuda::atomic<std::size_t, cuda::thread_scope_device>));
  *start_idx = 0;
}

template <typename K, typename V>
GPUHashTable<K, V>::~GPUHashTable() {
  delete hash_table;
  cudaFree(start_idx);
}

template <typename K, typename V>
int32 GPUHashTable<K, V>::Size() {
  return hash_table->map_.get_size();
}

template <typename K, typename V,
          typename CUCOAllocator = gpu_hash_map_tf_allocator<uint8_t>>
class StaticHashTable {
 public:
  cuco::static_map<K, int32, cuda::thread_scope_device, CUCOAllocator> map_;

  StaticHashTable(size_t initial_capacity, K empty_key_sentinel,
                  int32 empty_value_sentinel, CUCOAllocator alloc)
      : map_(initial_capacity, empty_key_sentinel, empty_value_sentinel,
             alloc) {}
};

template <typename K, typename V>
GPUStaticHashTable<K, V>::GPUStaticHashTable(size_t capacity, int dimension,
                                             K empty_key_sentinel,
                                             int32 empty_value_sentinel,
                                             Allocator* alloc,
                                             cudaStream_t stream) {
  capacity_ = capacity;
  dimension_ = dimension;
  // cudaMallocAsync(&values_d, sizeof(V) * dimension * capacity, stream);
  // cudaMallocManaged(&values_d, sizeof(V) * dimension * capacity);

  hash_table = new StaticHashTable<K, V>(
      capacity / 0.8 /*load_factor*/, empty_key_sentinel, empty_value_sentinel,
      gpu_hash_map_tf_allocator<uint8_t>(alloc));
}

template <typename K, typename V>
GPUStaticHashTable<K, V>::~GPUStaticHashTable() {
  delete hash_table;
  delete default_values;
  cudaFree(values_d);
}

template <typename K, typename V>
std::size_t GPUStaticHashTable<K, V>::Size() {
  return hash_table->map_.get_size();
}

#define REGISTER_ALL_TYPE(type)                   \
  template class GPUHashTable<int32, type>;       \
  template class GPUHashTable<int64, type>;       \
  template class GPUStaticHashTable<int32, type>; \
  template class GPUStaticHashTable<int64, type>;
TF_CALL_REAL_NUMBER_TYPES(REGISTER_ALL_TYPE)
#undef REGISTER_ALL_TYPE

namespace functor {
using atomicT = cuda::atomic<std::size_t, cuda::thread_scope_device>;

template <uint32_t block_size, uint32_t tile_size, typename Key, typename V,
          typename mutableViewT,
          typename Hash = cuco::detail::MurmurHash3_32<Key>,
          typename KeyEqual = thrust::equal_to<Key>>
__global__ void kv_initialize_static_map(const Key* key_first, int32 num_items,
                                         int32 dimension,
                                         mutableViewT map_mutable_view,
                                         atomicT* num_successes,
                                         Hash hash = Hash{},
                                         KeyEqual key_equal = KeyEqual{}) {
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;

  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  while (key_idx < num_items) {
    auto key = *(key_first + key_idx);
    int32 value = key_idx * dimension;

    auto const insert_pair = cuco::pair_type<Key, int32>{key, value};
    if (map_mutable_view.insert(tile, insert_pair, hash, key_equal) &&
        tile.thread_rank() == 0) {
      thread_num_successes++;
    }

    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
  std::size_t block_num_successes =
      BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    *num_successes += block_num_successes;
  }
}

template <typename Key, typename V>
struct KvInitStaticMap<GPUDevice, Key, V> {
  void operator()(const Key* keys, GPUStaticHashTable<Key, V>* hash_table,
                  int32 num_items, int32 dimension, cudaStream_t stream) {
    using MutableViewT = typename cuco::static_map<
        Key, int32, cuda::thread_scope_device,
        gpu_hash_map_tf_allocator<uint8_t>>::device_mutable_view;

    auto& map = hash_table->hash_table->map_;
    size_t num_to_insert = num_items;
    while (num_to_insert > 0) {
      static_assert(sizeof(std::size_t) == sizeof(atomicT));
      CUCO_CUDA_TRY(
          cudaMemsetAsync(map.get_num_success(), 0, sizeof(atomicT), stream));

      auto n = std::min((size_t)65535, num_to_insert);
      auto const grid_size =
          (TILE_SIZE * n + STRIDE * BLOCK_SIZE - 1) / (STRIDE * BLOCK_SIZE);
      TF_CHECK_OK(GpuLaunchKernel(
          kv_initialize_static_map<BLOCK_SIZE, TILE_SIZE, Key, V, MutableViewT,
                                   cuco::detail::MurmurHash3_32<Key>,
                                   thrust::equal_to<Key>>,
          grid_size, BLOCK_SIZE, 0, stream, keys, n, dimension,
          map.get_device_mutable_view(), map.get_num_success(),
          cuco::detail::MurmurHash3_32<Key>{}, thrust::equal_to<Key>{}));

      CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

      std::size_t h_num_successes =
          map.get_num_success()->load(cuda::std::memory_order_relaxed);
      map.update_size(h_num_successes);
      keys += n;
      num_to_insert -= n;
    }
  }
};

template <uint32_t block_size, uint32_t tile_size, typename Key, typename V,
          typename ViewT, typename Hash = cuco::detail::MurmurHash3_32<Key>,
          typename KeyEqual = thrust::equal_to<Key>>
__global__ void kv_lookup_dynamic_key_kernel(
    const Key* key_first, V** value_srcs, V* value_first, const V* default_v,
    int32 default_v_num, size_t num_items, int32 dimension, ViewT* submap_views,
    uint32_t num_submaps, int32 slot_idx, int32 slot_num, int32 bank_size,
    Hash hash = Hash{}, KeyEqual key_equal = KeyEqual{}) {
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();

  while (key_idx < num_items) {
    auto key = *(key_first + key_idx);
    int32 found_value = empty_value_sentinel;

    for (auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      auto found = submap_view.find(tile, key, hash, key_equal);
      if (found != submap_view.end()) {
        found_value = found->second;
        break;
      }
    }
    if (found_value == empty_value_sentinel) {
      for (int id = tile.thread_rank(); id < dimension; id += tile_size) {
        value_first[key_idx * dimension + id] =
            default_v[key % default_v_num * dimension + id];
      }
    } else {
      auto bank_idx = found_value / bank_size;
      auto offset_in_bank = found_value % bank_size;
      auto slot_offset = bank_idx * slot_num + slot_idx;
      for (int id = tile.thread_rank(); id < dimension; id += tile_size) {
        value_first[key_idx * dimension + id] =
            value_srcs[slot_offset][offset_in_bank * dimension + id];
      }
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}

template <typename Key, typename V>
struct KvLookupKey<GPUHashTable<Key, V>, Key, V> {
  void operator()(const Key* keys, V* vals, int32 num_items, int32 dimension,
                  int32 slot_idx, int32 slot_num,
                  GPUHashTable<Key, V>* hash_table, const V* default_v,
                  int32 default_v_num, cudaStream_t stream) {
    using mutableViewT = typename cuco::dynamic_map<
        Key, int32, cuda::thread_scope_device,
        gpu_hash_map_tf_allocator<uint8_t>>::mutable_view_type;
    using ViewT = typename cuco::dynamic_map<
        Key, int32, cuda::thread_scope_device,
        gpu_hash_map_tf_allocator<uint8_t>>::view_type;

    auto& map = hash_table->hash_table->map_;

    auto const grid_size = (TILE_SIZE * num_items + STRIDE * BLOCK_SIZE - 1) /
                           (STRIDE * BLOCK_SIZE);
    TF_CHECK_OK(GpuLaunchKernel(
        kv_lookup_dynamic_key_kernel<BLOCK_SIZE, TILE_SIZE, Key, V, ViewT>,
        grid_size, BLOCK_SIZE, 0, stream, keys, hash_table->d_bank_ptrs, vals,
        default_v, default_v_num, num_items, dimension,
        map.get_submap_views().data().get(), map.get_submaps().size(), slot_idx,
        slot_num, hash_table->initial_bank_size,
        cuco::detail::MurmurHash3_32<Key>{}, thrust::equal_to<Key>{}));
  }
};

template <uint32_t block_size, uint32_t tile_size, typename Key, typename V,
          typename ViewT, typename Hash = cuco::detail::MurmurHash3_32<Key>,
          typename KeyEqual = thrust::equal_to<Key>>
__global__ void kv_lookup_static_key_kernel(const Key* key_first,
                                            const V* value_srcs, V* value_first,
                                            const V* default_v, int32 default_v_num,
                                            size_t num_items, int32 dimension,
                                            ViewT map_views, Hash hash = Hash{},
                                            KeyEqual key_equal = KeyEqual{}) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);

  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;  // actual thread idx
  auto empty_value_sentinel = map_views.get_empty_value_sentinel();

  while (key_idx < num_items) {
    auto key = *(key_first + key_idx);
    int32 found_value = empty_value_sentinel;
    auto found = map_views.find(tile, key, hash, key_equal);
    if (found != map_views.end()) {
      found_value = found->second;
    }

    if (found_value == empty_value_sentinel) {
      for (int id = tile.thread_rank(); id < dimension; id += tile_size) {
        value_first[key_idx * dimension + id] =
            default_v[key % default_v_num * dimension + id];
      }
    } else {
      for (int id = tile.thread_rank(); id < dimension; id += tile_size) {
        value_first[key_idx * dimension + id] = value_srcs[found_value + id];
      }
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}

template <typename Key, typename V>
struct KvLookupKey<GPUStaticHashTable<Key, V>, Key, V> {
  void operator()(const Key* keys, V* vals, int32 num_items, int32 dimension,
                  int32 slot_idx, int32 slot_num,
                  GPUStaticHashTable<Key, V>* hash_table, const V* default_v,
                  int32 default_v_num, cudaStream_t stream) {
    using ViewT = typename cuco::static_map<
        Key, int32, cuda::thread_scope_device,
        gpu_hash_map_tf_allocator<uint8_t>>::device_view;
    auto& map = hash_table->hash_table->map_;

    auto const grid_size = (TILE_SIZE * num_items + STRIDE * BLOCK_SIZE - 1) /
                           (STRIDE * BLOCK_SIZE);
    TF_CHECK_OK(GpuLaunchKernel(
        kv_lookup_static_key_kernel<BLOCK_SIZE, TILE_SIZE, Key, V, ViewT>,
        grid_size, BLOCK_SIZE, 0, stream, keys, hash_table->values_d, vals,
        default_v, default_v_num, num_items, dimension, map.get_device_view(),
        cuco::detail::MurmurHash3_32<Key>{}, thrust::equal_to<Key>{}));
  }
};

template <uint32_t block_size, uint32_t tile_size, typename Key,
          typename mutableViewT, typename ViewT,
          typename Hash = cuco::detail::MurmurHash3_32<Key>,
          typename KeyEqual = thrust::equal_to<Key>>
__global__ void kv_lookup_and_insert_key_kernel(
    const Key* key_first, int32* value_first, int32 num_items,
    mutableViewT* submap_mutable_views, ViewT* submap_views,
    uint32_t num_submaps, atomicT* num_successes, atomicT* start_idx,
    int32 submap_idx, Hash hash = Hash{}, KeyEqual key_equal = KeyEqual{}) {
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;

  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();
  int32 tmp;

  while (key_idx < num_items) {
    auto key = *(key_first + key_idx);
    int32 found_value = empty_value_sentinel;

    for (auto i = 0; i < num_submaps; ++i) {
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
      auto insert_pair = cuco::pair_type<Key, int32>{key, found_value};
      if (submap_mutable_views[submap_idx].insert(tile, insert_pair, hash,
                                                  key_equal) &&
          tile.thread_rank() == 0) {
        thread_num_successes++;
      }
    }

    if (tile.thread_rank() == 0) {
      *(value_first + key_idx) = found_value;
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }

  std::size_t block_num_successes =
      BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    *num_successes += block_num_successes;
  }
}

template <typename Key, typename V>
struct KvLookupInsertKey<GPUDevice, Key, V> {
  void operator()(const Key* key_first, int32* value_first, int32 num_items,
                  GPUHashTable<Key, V>* hash_table, atomicT* start_idx,
                  cudaStream_t stream) {
    using mutableViewT = typename cuco::dynamic_map<
        Key, int32, cuda::thread_scope_device,
        gpu_hash_map_tf_allocator<uint8_t>>::mutable_view_type;
    using ViewT = typename cuco::dynamic_map<
        Key, int32, cuda::thread_scope_device,
        gpu_hash_map_tf_allocator<uint8_t>>::view_type;
    auto& map = hash_table->hash_table->map_;
    map.reserve(map.get_size() + num_items);
    uint32_t submap_idx = 0;
    std::size_t num_to_insert = num_items;

    while (num_to_insert > 0) {
      std::size_t capacity_remaining =
          map.get_max_load_factor() *
              map.get_submaps()[submap_idx]->get_capacity() -
          map.get_submaps()[submap_idx]->get_size();
      if (capacity_remaining >= map.get_min_insert_size()) {
        *(map.get_num_successes()) = 0;
        int device_id;
        CUCO_CUDA_TRY(cudaGetDevice(&device_id));
        CUCO_CUDA_TRY(cudaMemPrefetchAsync(map.get_num_successes(),
                                           sizeof(atomicT), device_id));

        auto n = std::min(capacity_remaining, num_to_insert);

	auto const grid_size = (TILE_SIZE * n + STRIDE * BLOCK_SIZE - 1) /
                           (STRIDE * BLOCK_SIZE);
        TF_CHECK_OK(GpuLaunchKernel(
            kv_lookup_and_insert_key_kernel<
                BLOCK_SIZE, TILE_SIZE, Key, mutableViewT, ViewT,
                cuco::detail::MurmurHash3_32<Key>, thrust::equal_to<Key>>,
            grid_size, BLOCK_SIZE, 0, stream, key_first, value_first, n,
            map.get_submap_mutable_views().data().get(),
            map.get_submap_views().data().get(), map.get_submaps().size(),
            map.get_num_successes(), start_idx, submap_idx,
            cuco::detail::MurmurHash3_32<Key>{}, thrust::equal_to<Key>{}));
        CUCO_CUDA_TRY(cudaDeviceSynchronize());
        std::size_t h_num_successes =
            map.get_num_successes()->load(cuda::std::memory_order_relaxed);
        map.update_submap_sizes(submap_idx, h_num_successes);
        key_first += n;
        value_first += n;
        num_to_insert -= n;
      }
      submap_idx++;
    }
  }
};

template <typename Key, typename Value>
__global__ void kv_lookup_or_create_emb_kernel(
    const Key* key_first, Value* val, Value* default_v, int64 dim,
    int32* item_idxs, int32 slot_idx, Value** d_banks, 
    bool** d_flags, int32 slot_num, int32 default_v_num, int32 bank_size) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto slot_offset = bank_idx * slot_num + slot_idx;
  bool stored = d_flags[slot_offset][offset_in_bank];
  __syncthreads();
  if (stored == false) {
    d_flags[slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      int32 default_v_idx = *(key_first + item_idx) % default_v_num;
      d_banks[slot_offset][offset_in_bank * dim + id] =
          default_v[default_v_idx * dim + id];
    }
  }
  for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
    val[item_idx * dim + id] = d_banks[slot_offset][offset_in_bank * dim + id];
  }
}

template <typename Key, typename Value>
struct KvLookupCreateEmb<GPUDevice, Key, Value> {
  void operator()(const Key* key_first, Value* val, Value* default_v, int64 dim,
                  int32* item_idxs, int32 num_items, int32 slot_idx,
                  int32 default_v_num,
                  Value** d_banks, bool** d_flags, int32 slot_num,
                  int32 bank_size, cudaStream_t stream) {
    auto const block_size = 256;
    auto const grid_size = num_items;
    TF_CHECK_OK(
        GpuLaunchKernel(kv_lookup_or_create_emb_kernel<Key, Value>, grid_size,
                        block_size, 0, stream, key_first, val, default_v, dim,
                        item_idxs, slot_idx,
                        d_banks, d_flags, slot_num, default_v_num, bank_size));
  }
};

template <typename Key, typename Value>
__global__ void kv_update_emb_kernel(const Key* key_first, Value* default_v,
                                     int64 dim, int32* item_idxs,
                                     int32 slot_idx, Value** d_banks,
                                     bool** d_flags, int32 slot_num,
                                     int32 default_v_num, int32 bank_size) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto slot_offset = bank_idx * slot_num + slot_idx;
  bool stored = d_flags[slot_offset][offset_in_bank];
  __syncthreads();
  if (stored == false) {
    d_flags[slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      int32 default_v_idx;
      default_v_idx = item_idx % default_v_num;
      d_banks[slot_offset][offset_in_bank * dim + id] =
          default_v[default_v_idx * dim + id];
    }
  }
}

template <typename Key, typename Value>
struct KvUpdateEmb<GPUDevice, Key, Value> {
  void operator()(const Key* key_first, Value* default_v, int64 dim,
                  int32* item_idxs, int32 num_items, int32 slot_idx,
                  int32 default_v_num, Value** d_banks, bool** d_flags,
                  int32 slot_num, int32 bank_size, cudaStream_t stream) {
    auto const block_size = 256;
    auto const grid_size = num_items;
    TF_CHECK_OK(GpuLaunchKernel(kv_update_emb_kernel<Key, Value>, grid_size,
                                block_size, 0, stream, key_first, default_v,
                                dim, item_idxs, slot_idx, d_banks, d_flags,
                                slot_num, default_v_num, bank_size));
  }
};

template <typename Key, typename ViewT,
          typename Hash = cuco::detail::MurmurHash3_32<Key>,
          typename KeyEqual = thrust::equal_to<Key>>
__global__ void kv_get_key_snapshot_kernel(
    Key* key, int32* item_idxs, int32 slot_idx, int32 primary_slot_idx,
    bool** d_flags, int32 bank_num, int32 slot_num, int32 bank_size,
    ViewT* submap_views, uint32_t num_submaps, int32 ev_size,
    Hash hash = Hash{}, KeyEqual key_equal = KeyEqual{}) {
  int n = 0;
  for (auto i = 0; i < num_submaps; ++i) {
    auto submap_view_size = submap_views[i].get_capacity();
    for (auto j = 0; j < submap_view_size; ++j) {
      auto found = submap_views[i].get_slot(j, hash, key_equal);
      if (found != submap_views[i].end()) {
        int32 item_pos = found->second;
        auto bank_idx = item_pos / bank_size;
        auto offset_in_bank = item_pos % bank_size;
        auto slot_offset = bank_idx * slot_num + slot_idx;
        auto pri_slot_offset = bank_idx * slot_num + primary_slot_idx;
        if (d_flags[slot_offset][offset_in_bank] &&
            d_flags[pri_slot_offset][offset_in_bank]) {
          *(key + n) = found->first;
          *(item_idxs + n) = found->second;
          ++n;
        }
      }
    }
  }
  for (auto i = n; i < ev_size; ++i) {
    *(key + n) = submap_views[0].get_empty_key_sentinel();
  }
}

template <typename Key, typename V>
struct KvKeyGetSnapshot<GPUDevice, Key, V> {
  void operator()(Key* key_first, int32* value_first, int32 slot_idx,
                  int32 primary_slot_idx, bool** d_flags, int32 bank_num,
                  int32 slot_num, int32 bank_size,
                  GPUHashTable<Key, V>* hash_table, int32 ev_size,
                  cudaStream_t stream) {
    using ViewT = typename cuco::dynamic_map<
        Key, int32, cuda::thread_scope_device,
        gpu_hash_map_tf_allocator<uint8_t>>::view_type;
    auto& map = hash_table->hash_table->map_;

    auto const block_size = 1;
    auto const grid_size = 1;
    TF_CHECK_OK(GpuLaunchKernel(
        kv_get_key_snapshot_kernel<Key, ViewT,
                                   cuco::detail::MurmurHash3_32<Key>,
                                   thrust::equal_to<Key>>,
        grid_size, block_size, 0, stream, key_first, value_first, slot_idx,
        primary_slot_idx, d_flags, bank_num, slot_num, bank_size,
        map.get_submap_views().data().get(), map.get_submaps().size(), ev_size,
        cuco::detail::MurmurHash3_32<Key>{}, thrust::equal_to<Key>{}));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
  }
};

template <typename Key, typename Value>
__global__ void kv_emb_get_snapshot_kernel(Key* key, Value* val,
                                           Key empty_key_sentinel, int64 dim,
                                           int32* item_idxs, int32 slot_idx,
                                           Value** d_banks, int32 bank_num,
                                           int32 slot_num, int32 bank_size,
                                           int32 total_num) {
  auto item_idx = blockIdx.x;
  if (item_idx < total_num) {
    auto item_pos = item_idxs[item_idx];
    auto bank_idx = item_pos / bank_size;
    auto offset_in_bank = item_pos % bank_size;
    auto slot_offset = bank_idx * slot_num + slot_idx;
    if (key[item_idx] != empty_key_sentinel) {
      for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
        val[item_idx * dim + id] =
            d_banks[slot_offset][offset_in_bank * dim + id];
      }
    }
  }
}

template <typename Key, typename Value>
struct KvEmbGetSnapshot<GPUDevice, Key, Value> {
  void operator()(Key* key, Value* val, Key empty_key_sentinel, int64 dim,
                  int32* item_idxs, int32 num_items, int32 slot_idx,
                  Value** d_banks, int32 bank_num, int32 slot_num,
                  int32 bank_size, cudaStream_t stream) {
    auto const block_size = 256;
    auto const grid_size = num_items;
    if (grid_size == 0) return;
    TF_CHECK_OK(GpuLaunchKernel(
        kv_emb_get_snapshot_kernel<Key, Value>, grid_size, block_size, 0,
        stream, key, val, empty_key_sentinel, dim, item_idxs, slot_idx, d_banks,
        bank_num, slot_num, bank_size, num_items));
  }
};

}  // namespace functor

#define REGISTER_ALL_TYPE(type)                                                \
  template struct functor::KvInitStaticMap<GPUDevice, int32, type>;            \
  template struct functor::KvInitStaticMap<GPUDevice, int64, type>;            \
  template struct functor::KvLookupInsertKey<GPUDevice, int32, type>;          \
  template struct functor::KvLookupInsertKey<GPUDevice, int64, type>;          \
  template struct functor::KvLookupCreateEmb<GPUDevice, int32, type>;          \
  template struct functor::KvLookupCreateEmb<GPUDevice, int64, type>;          \
  template struct functor::KvKeyGetSnapshot<GPUDevice, int32, type>;           \
  template struct functor::KvKeyGetSnapshot<GPUDevice, int64, type>;           \
  template struct functor::KvEmbGetSnapshot<GPUDevice, int32, type>;           \
  template struct functor::KvEmbGetSnapshot<GPUDevice, int64, type>;           \
  template struct functor::KvUpdateEmb<GPUDevice, int32, type>;                \
  template struct functor::KvUpdateEmb<GPUDevice, int64, type>;
TF_CALL_REAL_NUMBER_TYPES(REGISTER_ALL_TYPE)

#define REGISTER_LOOKUP_KERNEL_ALL(hash_table, type)                     \
  template struct functor::KvLookupKey<hash_table<int32, type>, int32, type>; \
  template struct functor::KvLookupKey<hash_table<int64, type>, int64, type > ;
#define REGISTER_INFERENCE_LOOKUP_KERNEL(type) \
  REGISTER_LOOKUP_KERNEL_ALL(GPUHashTable, type)
#define REGISTER_TRAINING_LOOKUP_KERNEL(type) \
  REGISTER_LOOKUP_KERNEL_ALL(GPUStaticHashTable, type)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_INFERENCE_LOOKUP_KERNEL)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_TRAINING_LOOKUP_KERNEL)

#undef REGISTER_INFERENCE_LOOKUP_KERNEL
#undef REGISTER_TRAINING_LOOKUP_KERNEL
#undef REGISTER_LOOKUP_KERNEL_ALL_TYPE
#undef REGISTER_ALL_TYPE

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

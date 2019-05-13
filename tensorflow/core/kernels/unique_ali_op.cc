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

#include <algorithm>
#include <limits>
#include <functional>
#include <unordered_map>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "sparsehash/dense_hash_map"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {
const int64 kPartitionLimit = 14336;
const int64 kPartitionSize = 8192;
const char* kUniqueOpHashMapEnv = "PAITF_UNIQUE_OP_HASH_MAP";
const char* kUniqueOpSerialEnv = "PAITF_UNIQUE_OP_SERIAL";
const char* kStlHashMapString = "STL";
const char* kAbslHashMapString = "ABSL";
const char* kGoogleHashMapString = "GOOGLE";

template <typename T>
const T InvalidHashKey() {
  return std::numeric_limits<T>::max();
}

template <class HashMap>
struct HashMapInitializer {
  static void InitSize(HashMap* hash_map, int64 capacity) {
    hash_map->reserve(2 * capacity);
  }
};

template <typename K, typename V>
struct HashMapInitializer<google::dense_hash_map<K, V>> {
  static void InitSize(google::dense_hash_map<K, V>* hash_map, int64 capacity) {
    hash_map->set_empty_key(InvalidHashKey<K>());
    hash_map->resize(2 * capacity);
  }
};

struct Range {
 public:
  explicit Range(int64 start, int64 end) : start_(start), end_(end) {}
  inline const int64 Start() const { return start_; }
  inline const int64 End() const { return end_; }
  inline const int64 Size() const { return end_ - start_; }
 private:
  const int64 start_, end_;
};

struct Partitioner {
 public:
  explicit Partitioner(int64 work_size, int32 num_parts) {
    if (work_size <= 0 || num_parts <= 0) { return; }
    num_parts_ = num_parts;
    parts_.reserve(num_parts);
    int64 start = 0;
    for (int32 i = 0; i < num_parts; ++i) {
      int64 end = start + (work_size + i) / num_parts;
      parts_.emplace_back(Range(start, end));
      start = end;
    }
  }

  const Range* GetRange(const int32 id) const {
    if (id < 0 || id >= num_parts_) { return nullptr; }
    return &parts_[id];
  }

  bool LocatePos(const int64 pos, int32* task_id) const {
    for (int32 i = 0; i < num_parts_; ++i) {
      if (pos >= parts_[i].Start() && pos < parts_[i].End()) {
        *task_id = i;
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<Range> parts_;
  int32 num_parts_ = 0;
};

class TaskRunner {
 public:
  explicit TaskRunner(const std::function<void(int32, int32)>& f,
                      thread::ThreadPool* tp, int32 n)
      : func_(f), thread_pool_(tp), num_tasks_(n) {}

  void Run() {
    if (num_tasks_ <= 0) return;
    BlockingCounter bc(num_tasks_ - 1);

    // Sending (num_tasks - 1) tasks to threadpool for scheduling
    for (int32 i = 0; i < num_tasks_ - 1; ++i) {
      thread_pool_->Schedule([this, &bc, i]() {
        func_(i, num_tasks_);
        bc.DecrementCount();
      });
    }
    // Run the last task in current thread.
    func_(num_tasks_ - 1, num_tasks_);
    bc.Wait();
  }

 private:
  std::function<void(int32 task_id, int32 num_tasks)> func_;
  thread::ThreadPool* thread_pool_;
  const int32 num_tasks_;
};

}  // anonymous namespace

template <typename T, typename TIndex>
class UniqueAliOp : public OpKernel {
 public:
  typedef absl::flat_hash_map<T, TIndex> DefaultHashMap;

  explicit UniqueAliOp(OpKernelConstruction* context) : OpKernel(context) {
    auto comp_with_stl_hash_map = [this] (OpKernelContext* context) {
      ComputeInternal<std::unordered_map<T, TIndex>>(context);
    };
    auto comp_with_absl_hash_map = [this] (OpKernelContext* context) {
      ComputeInternal<absl::flat_hash_map<T, TIndex>>(context);
    };
    auto comp_with_google_hash_map = [this] (OpKernelContext* context) {
      ComputeInternal<google::dense_hash_map<T, TIndex>>(context);
    };
    // NOTE(zycao>: Hash map insertion and lookup performance is dominating in
    // Unique Op. Based on benchmark results, 'google::dense_hash_map' will be
    // used as default for most key types except string, which will use
    // 'absl::flat_hash_map'.
    //
    // By setting "PAITF_UNIQUE_OP_HASH_MAP" environment variable, a particular
    // hash map could be seleteed to use. Possible choices are listed below:
    //     "STL" for std::unordred_map,
    //     "ABSL" for absl::flat_hash_map,
    //     "GOOGLE" for google::dense_hash_map.
    string hash_map_str;
    const char* hash_map_env = getenv(kUniqueOpHashMapEnv);
    if (hash_map_env) {
      hash_map_str = string(hash_map_env);
      std::transform(hash_map_str.begin(), hash_map_str.end(),
                     hash_map_str.begin(), ::toupper);
      hash_map_assigned_ = true;
    }
    bool is_string = std::is_same<T, string>::value;
    if (!hash_map_str.compare(kStlHashMapString)) {
      compute_internal_ = comp_with_stl_hash_map;
    } else if (!hash_map_str.compare(kGoogleHashMapString)) {
      compute_internal_ = comp_with_google_hash_map;
    } else if (!hash_map_str.compare(kAbslHashMapString) || is_string) {
      compute_internal_ = comp_with_absl_hash_map;
    } else {
      compute_internal_ = comp_with_google_hash_map;
      hash_map_assigned_ = false;
    }
  }

  void Compute(OpKernelContext* context) override {
    compute_internal_(context);
  }

 private:
  template<class HashMap>
  void ComputeInternal(OpKernelContext* context) {
    const Tensor& input = context->input(0);
    // TODO(dga):  Make unique polymorphic for returning int32 and int64
    // vectors to support large tensors.
    OP_REQUIRES(context,
                input.NumElements() <= std::numeric_limits<int32>::max(),
                errors::InvalidArgument(
                    "unique does not support input tensors larger than ",
                    std::numeric_limits<int32>::max(), " elements"));

    int64 axis = 0;
    std::vector<int64> new_sizes{1, input.NumElements(), 1};
    if (context->num_inputs() == 1) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                  errors::InvalidArgument("unique expects a 1D vector."));
    } else {
      // In case of UniqueV2, the axis is a 1D vector. The purpose is
      // to allow specifying either "no axis" or "axis". The `[]` means
      // "no axis", while `[x]` means `axis = x`.
      const Tensor& axis_tensor = context->input(1);
      OP_REQUIRES(context, TensorShapeUtils::IsVector(axis_tensor.shape()),
                  errors::InvalidArgument("axis expects a 1D vector."));
      OP_REQUIRES(
          context, axis_tensor.NumElements() <= 1,
          errors::InvalidArgument(
              "axis does not support input tensors larger than 1 elements"));
      if (axis_tensor.NumElements() == 0) {
        OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                    errors::InvalidArgument("unique expects a 1D vector."));
      } else {
        OP_REQUIRES(context,
                    (axis_tensor.dtype() == DT_INT32 ||
                     axis_tensor.dtype() == DT_INT64),
                    errors::InvalidArgument(
                        "axis tensor should be int32 or int64, but got ",
                        DataTypeString(axis_tensor.dtype())));
        if (axis_tensor.dtype() == DT_INT32) {
          axis = internal::SubtleMustCopy(axis_tensor.scalar<int32>()());
        } else {
          axis = internal::SubtleMustCopy(axis_tensor.scalar<int64>()());
        }
        axis = axis < 0 ? axis + input.dims() : axis;
        OP_REQUIRES(context, 0 <= axis && axis < input.dims(),
                    errors::InvalidArgument("axis has to be between [0, ",
                                            input.dims(), ")"));
        if (axis > 0) {
          for (int64 i = 0; i < axis; i++) {
            new_sizes[0] *= input.dim_size(i);
          }
        }
        new_sizes[1] = input.dim_size(axis);
        if (axis + 1 < input.dims()) {
          for (int64 i = axis + 1; i < input.dims(); i++) {
            new_sizes[2] *= input.dim_size(i);
          }
        }
      }
    }

    Tensor* idx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({new_sizes[1]}), &idx));
    auto idx_vec = idx->template vec<TIndex>();

    int64 uniq_size;
    if (new_sizes[0] == 1 && new_sizes[2] == 1) {
      // Specialized and faster implementation when unique is run over single
      // elements. Here we put T directly into the map rather than ints pointing
      // to them as in the general case.
      auto Tin = input.flat<T>();
      const int64 N = static_cast<int64>(Tin.size());

      // Calling PAI-TF UniqueV1 implementation.
      if (N >= kPartitionLimit && getenv(kUniqueOpSerialEnv) == nullptr) {
        ParallelComputeV1<HashMap>(context, input, idx, axis, &uniq_size);
      } else {
        if (hash_map_assigned_) {
          SerialComputeV1<HashMap>(context, input, idx, axis, &uniq_size);
        } else {
          SerialComputeV1<DefaultHashMap>(context, input, idx, axis,
                                          &uniq_size);
        }
      }
    } else {
      // General implementation when unique is run over multiple elements.
      auto Tin = input.shaped<T, 3>(new_sizes);

      auto hash_fn = [&Tin](const int64& key) {
        size_t h = 0;
        for (int64 i = 0; i < Tin.dimension(0); i++) {
          for (int64 j = 0; j < Tin.dimension(2); j++) {
            h = Hash64Combine(h, hash<T>{}(Tin(i, key, j)));
          }
        }
        return h;
      };

      auto equal_to_fn = [&Tin](const int64& lhs, const int64& rhs) {
        for (int64 i = 0; i < Tin.dimension(0); i++) {
          for (int64 j = 0; j < Tin.dimension(2); j++) {
            if (Tin(i, lhs, j) != Tin(i, rhs, j)) {
              return false;
            }
          }
        }
        return true;
      };

      std::unordered_map<int64, int64, decltype(hash_fn), decltype(equal_to_fn)>
          uniq(0, hash_fn, equal_to_fn);

      uniq.reserve(2 * Tin.dimension(1));

      for (int64 i = 0, j = 0; i < Tin.dimension(1); ++i) {
        auto it = uniq.insert(std::make_pair(i, j));
        idx_vec(i) = it.first->second;
        if (it.second) {
          ++j;
        }
      }

      uniq_size = static_cast<int64>(uniq.size());
      new_sizes[1] = uniq_size;
      TensorShape output_shape(input.shape());
      output_shape.set_dim(axis, uniq_size);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));
      auto Tout = output->shaped<T, 3>(new_sizes);

      for (auto it : uniq) {
        Tout.chip(it.second, 1) = Tin.chip(it.first, 1);
      }
    }

    if (num_outputs() > 2) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  2, TensorShape({uniq_size}), &output));
      auto count_output_vec = output->template vec<TIndex>();
      count_output_vec.setZero();
      const int N = idx_vec.size();
      for (int64 i = 0; i < N; ++i) {
        count_output_vec(idx_vec(i))++;
      }
    }
  }

  template<class HashMap>
  void SerialComputeV1(OpKernelContext* context, const Tensor& input,
                       Tensor* idx, int64 axis, int64* uniq_size) {
    auto Tin = input.flat<T>();
    const int64 N = input.NumElements();
    auto idx_vec = idx->template vec<TIndex>();

    HashMap uniq;
    HashMapInitializer<HashMap>::InitSize(&uniq, N);
    for (int64 i = 0, j = 0; i < N; ++i) {
      auto it = uniq.emplace(Tin(i), j);
      idx_vec(i) = it.first->second;
      if (it.second) {
        ++j;
      }
    }

    *uniq_size = static_cast<int64>(uniq.size());
    TensorShape output_shape(input.shape());
    output_shape.set_dim(axis, *uniq_size);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto Tout = output->flat<T>();

    for (auto it : uniq) {
      Tout(it.second) = it.first;
    }
  }

  template<class HashMap>
  void ParallelComputeV1(OpKernelContext* context, const Tensor& input,
                         Tensor* idx, int64 axis, int64* uniq_size) {
    // Struct INode was used to store an inverse mapping for each node in the
    // hash map container.
    struct INode {
      explicit INode(const TIndex index, const T& key)
          : owner_ptr_(nullptr), index_(index), key_(key) {}

      const INode* owner_ptr_;
      TIndex index_;
      const T key_;
    };

    // Struct UniqueSubMap is used to build and operate local hash map and keep
    // index mapping information.
    struct UniqueSubMap {
     public:
      inline void Init(int64 size) {
        next_index_ = 0;
        HashMapInitializer<HashMap>::InitSize(&uniq_, size);
        inodes_.reserve(size);
      }

      inline void UniqueInsert(const T& key) {
        auto it = uniq_.emplace(key, next_index_);
        if (it.second) {
          inodes_.emplace_back(INode(next_index_, key));
          ++next_index_;
        }
      }

      inline const INode* GetINodeByPos(const TIndex pos) const {
        const INode* inode = &inodes_[pos];
        return (inode->owner_ptr_ == nullptr) ? inode : inode->owner_ptr_;
      }

      inline const INode* GetINodeByKey(const T& key) const {
        auto item = uniq_.find(key);
        if (item != uniq_.end()) { return GetINodeByPos(item->second); }
        return nullptr;
      }

      bool DeDup(const TIndex pos, const UniqueSubMap& prior_map) {
        INode* my_inode = &inodes_[pos];
        if (my_inode->owner_ptr_ != nullptr) { return false; }
        const INode* prior_inode = prior_map.GetINodeByKey(my_inode->key_);
        if (prior_inode == nullptr) { return false; }
        my_inode->owner_ptr_ = prior_inode;
        return true;
      }

      bool TryIndexAndGetKey(const TIndex pos, const TIndex new_id, T* out) {
        INode* inode = &inodes_[pos];
        if (inode->owner_ptr_ != nullptr) { return false; }
        inode->index_ = new_id;
        *out = inode->key_;
        return true;
      }

      inline int64 Size() const { return static_cast<int64>(next_index_); }

     private:
      TIndex next_index_;
      HashMap uniq_;
      std::vector<INode> inodes_;
    };

    // NOTE(zycao): A four-step scheme is adopted for parallel unique computing.
    // Step 1: Seperate input data into T1 sections. build individual local hash
    //         maps M(0 .. (T1 - 1)) for each section.
    // Step 2: Mark and count duplicated keys accross all T1 hash maps. For each
    //         key stored in hasp map M(i), it needs to do lookups from hash map
    //         M(0) to M(i-1) to check possible duplicates. Thus keys stored in
    //         M(i, i = 1 .. (T1 - 1) would be divided into T2 parts, and then
    //         processed simultanously in T2 tasks.
    // Step 3: Calculate the global unique index for all keys, based on marking
    //         and counting result of Step 2. Hash maps would be processed by
    //         T1 tasks in parallel.
    // Step 4: Fill the output Tensor with multiple tasks as many as possible.
    //
    // Since the complexity of Step (1,3) and Step 2 would be affected by the
    // number of T1 in opposite direction. A simple deduction was done and it
    // indicates that ideal T1 size should be in the order of O(T2 ^ 1/3 * c).
    //     >>  T1_ideal ~= ((beta * max_threads) ^ 1/3) + 1/2
    // Here 'beta' is a factor used to approximately describe hash map lookup
    // speed compared to insert operations. This result is adopted in current
    // implemetation to decide Step 1 task size T1.
    auto Tin = input.flat<T>();
    const int64 N = input.NumElements();
    int32 max_threads =
      context->device()->tensorflow_cpu_worker_threads()->num_threads;
    auto thread_pool =
      context->device()->tensorflow_cpu_worker_threads()->workers;

    // Parallel Step 1: Build hash maps.
    const double factor = 10;  // Suppose lookup is 10x faster than insert.
    int32 max_tasks_t1
        = static_cast<int32>(std::cbrt(factor * max_threads) + 1);
    int32 num_tasks_t1 = std::max(std::min(max_threads, max_tasks_t1), 1);
    VLOG(1) << "[UniqueParallel] Step 1 num_tasks: " << num_tasks_t1;

    Partitioner map_parter(N, num_tasks_t1);
    std::vector<UniqueSubMap> uniq_maps(num_tasks_t1);

    auto MapBuildTask = [this, &Tin, &uniq_maps, &map_parter]
        (int32 task_id, int32 num_tasks) {
      UniqueSubMap& uniq_map = uniq_maps[task_id];
      const Range* range = map_parter.GetRange(task_id);
      uniq_map.Init(range->Size());
      for (int64 i = range->Start(); i < range->End(); ++i) {
        uniq_map.UniqueInsert(Tin(i));
      }
    };
    TaskRunner t1_runner(MapBuildTask, thread_pool, num_tasks_t1);
    t1_runner.Run();

    int64 est_dup_count_cost = 0;
    for (int32 i = 0; i < num_tasks_t1; ++i) {
      est_dup_count_cost += uniq_maps[i].Size() * i;
    }

    // Parallel Step 2: Check and count duplicated keys.
    int32 max_tasks_t2
        = (est_dup_count_cost + kPartitionSize - 1) / kPartitionSize;
    int32 num_tasks_t2 = std::max(std::min(max_threads, max_tasks_t2), 1);
    VLOG(1) << "[UniqueParallel] Step 2 num_tasks: " << num_tasks_t2;

    // Divide each of T1 hash maps into T2 parts, remember the offsets.
    std::vector<int64> dups(num_tasks_t1 * num_tasks_t2, 0);
    std::vector<Partitioner> dup_parters;
    dup_parters.reserve(num_tasks_t1);
    for (int32 i = 0; i < num_tasks_t1; ++i) {
      dup_parters.emplace_back(Partitioner(uniq_maps[i].Size(), num_tasks_t2));
    }

    auto DupCountTask = [this, &uniq_maps, &dups, &dup_parters, num_tasks_t1]
        (int32 task_id, int32 num_tasks) {
      // Using 3 layer loop to make all checks.
      for (int32 prior_id = 0; prior_id < num_tasks_t1 - 1; ++prior_id) {
        const UniqueSubMap& prior_map = uniq_maps[prior_id];
        for (int32 lat_id = prior_id + 1; lat_id < num_tasks_t1; ++lat_id) {
          UniqueSubMap& lat_map = uniq_maps[lat_id];
          int64 dup_offsets = lat_id * num_tasks;
          const Range* range = dup_parters[lat_id].GetRange(task_id);
          for (int64 i = range->Start(); i < range->End(); ++i) {
            if (lat_map.DeDup(i, prior_map)) { ++dups[dup_offsets + task_id]; }
          }
        }
      }
    };
    TaskRunner t2_runner(DupCountTask, thread_pool, num_tasks_t2);
    t2_runner.Run();

    // Calculate the global unique index numbers and global offset for every
    // hash map based on duplication checking results.
    std::vector<int64> global_offsets(num_tasks_t1, 0);
    for (int32 i = 0; i < num_tasks_t1 - 1; ++i) {
      global_offsets[i + 1] = global_offsets[i] + uniq_maps[i].Size();
      for (int32 j = 0; j < num_tasks_t2; ++j) {
        global_offsets[i + 1] -= dups[i * num_tasks_t2 + j];
      }
    }
    int64 num_tot_indices =
        global_offsets[num_tasks_t1 - 1] + uniq_maps[num_tasks_t1 - 1].Size();
    for (int32 j = 0; j < num_tasks_t2; ++j) {
      num_tot_indices -= dups[(num_tasks_t1 - 1) * num_tasks_t2 + j];
    }

    // Parallel Step 3: Recalculate global index for all keys in all hash maps.
    //                  Write the output keys Tensor at the same time.
    *uniq_size = num_tot_indices;
    TensorShape output_shape(input.shape());
    output_shape.set_dim(axis, num_tot_indices);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto key_output_vec = output->template vec<T>();

    auto GlobalIndexTask = [this, &key_output_vec, &uniq_maps, &global_offsets]
        (int32 task_id, int32 num_tasks) {
      TIndex cur_id = global_offsets[task_id];
      UniqueSubMap& uniq_map = uniq_maps[task_id];
      for (int64 i = 0; i < uniq_map.Size(); ++i) {
        if (uniq_map.TryIndexAndGetKey(i, cur_id, &key_output_vec(cur_id))) {
          ++cur_id;
        }
      }
    };
    TaskRunner t3_runner(GlobalIndexTask, thread_pool, num_tasks_t1);
    t3_runner.Run();

    // Parallel Step 4: Write output indicies Tensor.
    int32 max_tasks_t4 = (N + kPartitionSize - 1) / kPartitionSize;
    int32 num_tasks_t4 = std::max(std::min(max_threads, max_tasks_t4), 1);
    VLOG(1) << "[UniqueParallel] Step 4 num_tasks: " << num_tasks_t4;

    Partitioner fill_parter(N, num_tasks_t4);
    auto idx_vec = idx->template vec<TIndex>();

    auto OutputTask = [this, &Tin, &idx_vec, &uniq_maps, &fill_parter,
                       &map_parter] (int32 task_id, int32 num_tasks) {
      const Range* out_range = fill_parter.GetRange(task_id);
      int64 out_pos = out_range->Start();
      int32 map_id;
      if (!map_parter.LocatePos(out_pos, &map_id)) { return; }
      int64 map_range_end = map_parter.GetRange(map_id)->End();
      while (out_pos < out_range->End()) {
        const INode* inode = uniq_maps[map_id].GetINodeByKey(Tin(out_pos));
        idx_vec(out_pos) = inode->index_;
        ++out_pos;
        if (out_pos == map_range_end && out_pos < out_range->End()) {
          ++map_id;
          map_range_end = map_parter.GetRange(map_id)->End();
        }
      }
    };
    TaskRunner t4_runner(OutputTask, thread_pool, num_tasks_t4);
    t4_runner.Run();
  }

 private:
  std::function<void (OpKernelContext*)> compute_internal_;
  bool hash_map_assigned_ = false;
};

#define REGISTER_UNIQUE(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueAliOp<type, int32>);             \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueAliOp<type, int64>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueV2")                       \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueAliOp<type, int32>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueV2")                       \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueAliOp<type, int64>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueAliOp<type, int32>)              \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueAliOp<type, int64>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV2")             \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueAliOp<type, int32>)              \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV2")             \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueAliOp<type, int64>)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_UNIQUE);
REGISTER_UNIQUE(string)
REGISTER_UNIQUE(bool)
#undef REGISTER_UNIQUE

// Fake integer GPU kernels so that the use of Unique in optimizers (to
// de-duplicate sparse gradient indices) does not conflict with gradients being
// located on a GPU. These kernels run on the CPU, their inputs and outputs
// residing in host (not GPU) memory.
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int32, int32>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int32, int64>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("T")
                            .TypeConstraint<int32>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int64, int32>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("T")
                            .TypeConstraint<int64>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int64, int64>);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int32, int32>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int64>("T")
                            .TypeConstraint<int32>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int64, int32>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int32, int64>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int64>("T")
                            .TypeConstraint<int64>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int64, int64>);
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow

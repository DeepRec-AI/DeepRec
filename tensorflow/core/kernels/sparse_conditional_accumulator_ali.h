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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_CONDITIONAL_ACCUMULATOR_ALI_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_CONDITIONAL_ACCUMULATOR_ALI_H_

#include "sparsehash/dense_hash_map"
#include "tensorflow/core/kernels/task_runner.h"
#include "tensorflow/core/kernels/typed_conditional_accumulator_base.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/env_var.h"

#define unlikely(x) __builtin_expect(!!(x), 0)

namespace {
  static constexpr int kPartitionBlockSize = 65536;
  static constexpr int kMapBlockSize = 8192;
  static const int64_t kPreseverdEmptyKey =
    tensorflow::random::New64Configuable();
}

namespace tensorflow {

/* SUM: Reduction by sum of grads
 *  * MEAN_BY_COUNT: Reduction by sum/sparse_id_count
 *   * MEAN_BY_WORKER_COUNT: Reduction by sum/worker_count
 *    */
enum class GradReductionType {SUM, MEAN_BY_COUNT, MEAN_BY_WORKER_COUNT};

// "MultiMap" merges grads by multi maps in parallel each time
template <typename Device, typename T>
class SparseConditionalAccumulatorMultiMap
    : public TypedConditionalAccumulatorBase<
          std::tuple<const Tensor*, const Tensor*, const Tensor*>> {
 public:
  SparseConditionalAccumulatorMultiMap(const DataType& dtype,
                               const PartialTensorShape& shape,
                               const string& name, const string& reduction_type)
      : TypedConditionalAccumulatorBase<
            std::tuple<const Tensor*, const Tensor*, const Tensor*>>(
            dtype, shape, name, reduction_type) {
    if (reduction_type == "MEAN") {
      reduction_type_enum_ = GradReductionType::MEAN_BY_COUNT;
    } else if (reduction_type_ == "CMEAN") {
      reduction_type_enum_ = GradReductionType::MEAN_BY_WORKER_COUNT;
    }
    static char print_once = [] {
      LOG(INFO) << "SparseConditionalAccumulatorMultiMap preserved "
          "dense hash map key: " << kPreseverdEmptyKey;
      return '\0';
    }();
  }

  ~SparseConditionalAccumulatorMultiMap() override {
  };

 protected:
  typedef std::tuple<int, int, int> BIndex; // bid, off, cnt

  GradReductionType reduction_type_enum_ = GradReductionType::SUM;
  struct IdHash : public std::hash<int64> {
      inline std::size_t operator()(int64 const& i) const noexcept {
        size_t x = (i ^ (i >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
      }
  };
  std::vector<google::dense_hash_map<int64, BIndex, IdHash>> accum_grads_;
  int num_maps_{0};
  std::function<int64(int64)> key_to_table_idx_;
  std::vector<Tensor> bufs_;

  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>,
                           Eigen::Unaligned>
      SliceT;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                           Eigen::Unaligned>
      SliceConstT;

  void AllocateAndAssignToAccumGradFunction(
      OpKernelContext* ctx,
      std::tuple<const Tensor*, const Tensor*, const Tensor*>* grad) override {
    if (unlikely(accum_grads_.empty())) {
      // allocate accum_grads_ maps
      auto max_parallel = ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
      const Tensor* grad_idx = std::get<0>(*grad);
      num_maps_ = grad_idx->dim_size(0) / kMapBlockSize + 1;
      num_maps_ = std::min(num_maps_, max_parallel);
      for (int64 i = 0; i < num_maps_; ++i) {
        accum_grads_.emplace_back();
        accum_grads_.back().max_load_factor(0.7);
        accum_grads_.back().set_empty_key(kPreseverdEmptyKey);
      }
    }
    AccumGrads(ctx, grad);
  }

  void AddToAccumGradFunction(
      OpKernelContext* ctx,
      std::tuple<const Tensor*, const Tensor*, const Tensor*>* grad) override {
    AccumGrads(ctx, grad);
  }

  void DivideAccumGradByCounter(OpKernelContext* ctx) override
      EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    return;
  }

  bool SetOutput(OpKernelContext* ctx) override {
    return OutputGrads(ctx);
  }

  bool GetAndValidateTensorInputForApplyGrad(
      OpKernelContext* ctx,
      std::tuple<const Tensor*, const Tensor*, const Tensor*>** tensor) override
      EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    // TODO(xinghao, jmchen): The roundabout way of getting attr from
    // OpKernelContext (instead of OpKernelConstruction) is a hack, and should
    // be fixed if it affects efficiency.
    bool has_known_shape = false;
    OP_REQUIRES_OK_BOOLEAN(
        ctx, GetNodeAttr(ctx->op_kernel().def(), "has_known_shape",
                         &has_known_shape));

    // Get input gradient tensors
    const Tensor* grad_idx_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx,
                           ctx->input("gradient_indices", &grad_idx_tensor));
    const Tensor* grad_val_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx,
                           ctx->input("gradient_values", &grad_val_tensor));
    const Tensor* grad_shape_tensor = nullptr;
    if (has_known_shape) {
      OP_REQUIRES_OK_BOOLEAN(ctx,
                             ctx->input("gradient_shape", &grad_shape_tensor));
    }

    // Checks
    OP_REQUIRES_BOOLEAN(
        ctx, TensorShapeUtils::IsVector(grad_idx_tensor->shape()),
        errors::InvalidArgument(
            "Input indices should be vector but received shape: ",
            grad_idx_tensor->shape().DebugString()));
    const int64 nnz = grad_idx_tensor->dim_size(0);
    OP_REQUIRES_BOOLEAN(
        ctx, grad_val_tensor->dims() > 0,
        errors::InvalidArgument("Values cannot be 0-dimensional."));
    OP_REQUIRES_BOOLEAN(ctx, grad_val_tensor->dim_size(0) == nnz,
                        errors::InvalidArgument("Expected ", nnz,
                                                " non-empty input values, got ",
                                                grad_val_tensor->dim_size(0)));

    *tensor = new std::tuple<const Tensor*, const Tensor*, const Tensor*>(
        grad_idx_tensor, grad_val_tensor, grad_shape_tensor);

    return true;
  }

  void CleanUpGradTensor(std::tuple<const Tensor*, const Tensor*,
                                    const Tensor*>* tensor) override {
    if (tensor != nullptr) delete tensor;
  }

 private:

  void AccumGrads(OpKernelContext* ctx,
                   std::tuple<const Tensor*, const Tensor*, const Tensor*>* grad) {
    const Tensor* grad_idx = std::get<0>(*grad);
    const Tensor* grad_val = std::get<1>(*grad);

    const int64 nnz = grad_idx->dim_size(0);

    Tensor buf;
    ctx->allocate_temp(dtype_, grad_val->shape(), &buf);
    buf.flat<T>().device(ctx->template eigen_device<Device>()) = grad_val->flat<T>();
    auto buf_flat = buf.flat_outer_dims<T>();
    Eigen::DSizes<Eigen::DenseIndex, 1> slice_shape(buf_flat.dimension(1));
    size_t buf_idx = bufs_.size();
    bufs_.emplace_back(buf);
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    auto max_parallelism = ctx->device()->tensorflow_cpu_worker_threads()->num_threads;

    static IdHash hasher;
    int partition_threads = nnz / kPartitionBlockSize;
    partition_threads = std::min(partition_threads, max_parallelism);
    if (partition_threads == 0) partition_threads = 1;
    int64 partition_block_size = nnz / partition_threads;
    std::unique_ptr<int64[]> partitions{new int64[partition_threads * num_maps_]};
    std::unique_ptr<int64[]> id_chain{new int64[nnz]};
    auto PartitionTask = [this, &partitions, partition_block_size,
        &grad_idx, nnz, &id_chain] (int32 task_id, int32 num_tasks) {
      auto st = Status::OK();
      int64* cur_partition = partitions.get() + task_id * num_maps_;
      for (int64 i = 0; i < num_maps_; ++i) {
        *(cur_partition + i) = -1;
      }
      int64 lower_bound = task_id * partition_block_size;
      int64 upper_bound = (task_id + 1) * partition_block_size;
      if (task_id == num_tasks - 1) {
        upper_bound = std::max(upper_bound, nnz);
      }
      for (int64 i = lower_bound; i < upper_bound; ++i) {
        auto& idx = grad_idx->vec<int64>()(i);
        if (unlikely(idx == kPreseverdEmptyKey)) {
          st = errors::InvalidArgument(
              "Input id is preserved key of dense_hash_map, "
              "not supported: ", idx);
          break;
        }
        auto table_idx = (hasher(idx) >> 54) % num_maps_;
        id_chain[i] = *(cur_partition + table_idx);
        *(cur_partition + table_idx) = i;
      }
      return st;
    };
    SummaryTaskRunner<Status, StatusSummaryUpdater> t1_runner(
        PartitionTask, Status::OK(), thread_pool, partition_threads);
    t1_runner.Run();
    OP_REQUIRES_OK(ctx, t1_runner.summary());

    auto InsertGradTask = [this, nnz, &grad_idx, &grad_val, &buf_flat,
        &slice_shape, buf_idx, &partitions, partition_threads, &id_chain]
        (int32 task_id, int32 num_tasks) {
      int64* cur_partition = partitions.get();
      int64 cur_idx;
      for (int64 i = 0; i < partition_threads; ++i) {
        int64 next_idx = *(cur_partition + task_id);
        cur_partition += num_maps_;
        while(next_idx != -1) {
          cur_idx = next_idx;
          next_idx = id_chain[next_idx];
          auto key = grad_idx->vec<int64>()(cur_idx);
          auto& accum_grad = accum_grads_[task_id];
          auto it = accum_grad.find(key);
          if (it != accum_grad.end()) {
            auto& obidx = it->second;
            auto &obuf = bufs_[(int)std::get<0>(obidx)];
            auto obuf_flat = obuf.flat_outer_dims<T>();
            T* obuf_slice_ptr = &obuf_flat(std::get<1>(obidx), 0);
            SliceT obuf_slice(obuf_slice_ptr, slice_shape);

            T* buf_slice_ptr = &buf_flat(cur_idx, 0);
            SliceT buf_slice(buf_slice_ptr, slice_shape);

            buf_slice += obuf_slice;
            it->second = std::make_tuple(buf_idx, cur_idx, std::get<2>(obidx)+1);
          } else {
            accum_grad[key] = std::make_tuple(buf_idx, cur_idx, 1);
          }
        }
      }
    };
    if (nnz >= kPartitionBlockSize) {  // run in parallel
      TaskRunner t2_runner(InsertGradTask, thread_pool, num_maps_);
      t2_runner.Run();
    } else {  // run in current thread
      for (auto i = 0; i < num_maps_; ++i) {
        InsertGradTask(i, num_maps_);
      }
    }
  }

  bool OutputGrads(OpKernelContext* ctx) {
    Tensor* idx_tensor, *val_tensor;
    int64 nnz = 0;
    std::unique_ptr<int64[]> map_offsets{new int64[num_maps_]};
    for (size_t i = 0; i < num_maps_; ++i) {
      map_offsets[i] = nnz;
      nnz += accum_grads_[i].size();
    }
    auto val_shape = bufs_[0].shape();
    val_shape.set_dim(0, nnz);

    OP_REQUIRES_OK_BOOLEAN(ctx, ctx->allocate_output(0, {nnz}, &idx_tensor));
    OP_REQUIRES_OK_BOOLEAN(ctx, ctx->allocate_output(1, val_shape, &val_tensor));

    auto idx_tensor_vec = idx_tensor->vec<int64>();
    auto val_flat = val_tensor->flat_outer_dims<T>();
    Eigen::DSizes<Eigen::DenseIndex, 1> slice_shape(val_flat.dimension(1));

    auto OutputTask = [this, nnz, &map_offsets, &idx_tensor_vec, &val_shape,
            &val_flat, &slice_shape] (int32 task_id, int32 num_tasks) {
      int64 offset = map_offsets[task_id];
      for (auto iter = accum_grads_[task_id].begin();
          iter != accum_grads_[task_id].end(); ++iter) {
        auto &bidx = iter->second;
        auto &buf = bufs_[(int)std::get<0>(bidx)];
        auto buf_flat = buf.flat_outer_dims<T>();
        T* buf_slice_ptr = &buf_flat(std::get<1>(bidx), 0);
        SliceT buf_slice(buf_slice_ptr, slice_shape);

        T* val_slice_ptr = &val_flat(offset, 0);
        SliceT val_slice(val_slice_ptr, slice_shape);

        idx_tensor_vec(offset) = iter->first;
        if (reduction_type_enum_ == GradReductionType::MEAN_BY_COUNT) {
          val_slice = buf_slice / (T)(std::get<2>(bidx));
        } else if (reduction_type_enum_ == GradReductionType::MEAN_BY_WORKER_COUNT) {
          val_slice = buf_slice / (T)(counter_);
        } else {
          val_slice = buf_slice;
        }
        ++offset;
      }
    };
    if (nnz >= kPartitionBlockSize) {  // run in parallel
      auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
      TaskRunner t1_runner(OutputTask, thread_pool, num_maps_);
      t1_runner.Run();
    } else {
      for (auto i = 0; i < num_maps_; ++i) {
        OutputTask(i, num_maps_);
      }
    }

    int64 accum_val_dims = val_tensor->dims();
    Tensor* shape_tensor;
    OP_REQUIRES_OK_BOOLEAN(
        ctx, ctx->allocate_output(2, {accum_val_dims}, &shape_tensor));
    // If allocate_output fails, OP_REQUIRES_OK_BOOLEAN will short-circuit
    // the remaining code and just return false

    // First dim of shape is defined by shape_, others by accum_val_->shape
    shape_tensor->flat<int64>()(0) =
        (shape_.dims() > 0) ? shape_.dim_size(0) : -1;
    for (int64 i = 1; i < accum_val_dims; i++) {
      shape_tensor->flat<int64>()(i) = val_tensor->dim_size(i);
    }

    for (auto &accum_grad : accum_grads_) {
      accum_grad.clear();
    }
    bufs_.clear();

    return true;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SparseConditionalAccumulatorMultiMap);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_CONDITIONAL_ACCUMULATOR_ALI_H_

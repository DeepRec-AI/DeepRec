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

namespace tensorflow {

constexpr int64 DEFAULT_RESTORE_THREAD_NUM = 4;

class KvRestoreThreadPool {
 public:
  KvRestoreThreadPool() {
    TF_CHECK_OK(ReadInt64FromEnvVar("TF_EV_RESTORE_THREAD_NUM",
          DEFAULT_RESTORE_THREAD_NUM, &thread_num_));
  }

  static thread::ThreadPool* GetInstance() {
    static thread::ThreadPool tp(Env::Default(),
        "restore_ev_threadpool", thread_num_);
    return &tp;
  }

 private:
  static int64 thread_num_;
};

int64 KvRestoreThreadPool::thread_num_ =
    DEFAULT_RESTORE_THREAD_NUM;

template <typename TKey, typename TValue>
class KvResourceImportV2Op: public AsyncOpKernel {
 public:
  explicit KvResourceImportV2Op(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("counter_type", &counter_type_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("steps_to_live", &steps_to_live_));
    OP_REQUIRES(c, steps_to_live_ >= 0,
                 errors::InvalidArgument(
                    "steps_to_live must >= 0, ",
                    std::to_string(steps_to_live_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                 errors::InvalidArgument(
                    "partition_id must >= 0, ",
                    std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                 errors::InvalidArgument(
                    "partition_num must >= 1, ",
                    std::to_string(partition_num_)));
    //OP_REQUIRES_OK(c, c->GetAttr("restore_versions", &restore_versions_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_type", &ht_type_));
    OP_REQUIRES_OK(c, c->GetAttr("ht_partition_num", &ht_partition_num_));
    OP_REQUIRES_OK(c, c->GetAttr("emb_index", &emb_index_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_index", &slot_index_));
    OP_REQUIRES_OK(c, c->GetAttr("filter_freq", &filter_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("block_num", &block_num_));
    OP_REQUIRES_OK(c, c->GetAttr("max_element_size", &max_element_size_));
    OP_REQUIRES_OK(c, c->GetAttr("false_positive_probability",
          &false_positive_probability_));
    OP_REQUIRES_OK(c, c->GetAttr("l2_weight_threshold",
          &l2_weight_threshold_));
    OP_REQUIRES_OK(c, c->GetAttr("max_freq", &max_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_dim",
          &default_value_dim_));
    OP_REQUIRES_OK(c, c->GetAttr("default_value_no_permission",
          &default_value_no_permission_));
    OP_REQUIRES_OK(c, c->GetAttr("slot_num", &slot_num_));
    int64 storage_type = 0;
    OP_REQUIRES_OK(c, c->GetAttr("storage_type", &storage_type));
    storage_type_ = static_cast<embedding::StorageType>(storage_type);

    OP_REQUIRES_OK(c, c->GetAttr("storage_path", &storage_path_));
    OP_REQUIRES_OK(c, c->GetAttr("storage_size", &storage_size_));
    OP_REQUIRES_OK(c, c->GetAttr("record_freq", &record_freq_));
    OP_REQUIRES_OK(c, c->GetAttr("record_version", &record_version_));
    OP_REQUIRES_OK(c, c->GetAttr("reset_version", &reset_version_));

    if ((filter_freq_ != 0 && max_element_size_ == 0)
         || steps_to_live_ != -1 || record_freq_
         || record_version_ || storage_type > 5) {
      if (block_num_ > 1 || (filter_freq_ != 0 && storage_type <= 5)) {
        layout_ = "normal";
      } else {
        layout_ = "normal_contiguous";
      }
    } else {
      layout_ = "light";
    }

    CHECK(block_num_ == 1 || layout_ != "normal_contiguous");

    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EV_ASYNC_RESTORE", true,
                                   &ev_async_restore_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(4);
    const std::string name_string = name.scalar<string>()();
    const Tensor& default_values = context->input(3);
    OP_REQUIRES(context, dtype_ == default_values.dtype(),
        errors::InvalidArgument(
          "Variable and ddd value dtypes don't match; respectively, ",
          dtype_, " and ", default_values.dtype()));

    ResourceHandle handle_self = HandleFromInput(context, 1);
    ResourceHandle handle_primary = HandleFromInput(context, 2);
    std::string opname = handle_self.name();
    EmbeddingVar<TKey, TValue>* ev = nullptr;

    if (handle_self.name() == handle_primary.name() &&
         handle_self.container() == handle_primary.container()) {
      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, context,
             handle_self](EmbeddingVar<TKey, TValue>** ptr) {
             Allocator* allocator =
                context->device()->GetAllocator(AllocatorAttributes());
             auto embedding_config = EmbeddingConfig(
                 emb_index_ + block_num_ * slot_index_,
                 emb_index_,
                 block_num_, slot_num_, opname + "-primary",
                 steps_to_live_, filter_freq_,
                 max_freq_, l2_weight_threshold_,
                 layout_,  max_element_size_,
                 false_positive_probability_,
                 counter_type_, default_value_dim_,
                 default_value_no_permission_,
                 record_freq_, record_version_);
              auto storage_manager =
                new embedding::StorageManager<TKey, TValue>(
                  handle_self.name(), embedding::StorageConfig(
                    storage_type_, storage_path_, storage_size_, layout_,
                    embedding_config));
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                         storage_manager,
                         embedding_config,
                         allocator);
             return Status::OK();
            }));
      ev->Init(default_values, default_value_dim_);
    } else {
      EmbeddingVar<TKey, TValue>* primary_variable = nullptr;
      OP_REQUIRES_OK(
       context,
       LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
           context, handle_primary, &primary_variable,
           [this, default_values, opname, context,
            handle_primary](EmbeddingVar<TKey, TValue>** ptr) {
             int64 primary_slot_index(0), primary_emb_index(0);
             Allocator* allocator =
                context->device()->GetAllocator(AllocatorAttributes());
             auto embedding_config = EmbeddingConfig(
                 primary_emb_index + block_num_ * primary_slot_index,
                 primary_emb_index, block_num_, slot_num_,
                 opname + "-primary", steps_to_live_, filter_freq_,
                 max_freq_, l2_weight_threshold_,
                 layout_,  max_element_size_,
                 false_positive_probability_,
                 counter_type_, 0, record_freq_, record_version_);
             auto storage_manager =
               new embedding::StorageManager<TKey, TValue>(
                 handle_primary.name(), embedding::StorageConfig(
                   storage_type_, storage_path_, storage_size_,
                   layout_, embedding_config));
             *ptr = new EmbeddingVar<TKey, TValue>(handle_primary.name(),
                 storage_manager, embedding_config, allocator);
            // default_values is slot value, should not to initialize primary value
            return Status::OK();
           }));

      OP_REQUIRES_OK(
        context,
        LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
            context, handle_self, &ev,
            [this, default_values, opname, primary_variable,
             handle_self, context](EmbeddingVar<TKey, TValue>** ptr) {
              Allocator* allocator =
                context->device()->GetAllocator(AllocatorAttributes());
              *ptr = new EmbeddingVar<TKey, TValue>(handle_self.name(),
                  primary_variable->storage_manager(),
                  EmbeddingConfig(emb_index_ + block_num_ * slot_index_,
                    emb_index_, block_num_, slot_num_, opname,
                    steps_to_live_, filter_freq_, max_freq_,
                    l2_weight_threshold_, layout_, max_element_size_,
                    false_positive_probability_,
                    counter_type_, default_value_dim_,
                    default_value_no_permission_,
                    record_freq_, record_version_),
                    allocator);
             return (*ptr)->Init(default_values, default_value_dim_);
            }));
      core::ScopedUnref unref_me(primary_variable);
    }
    core::ScopedUnref unref_me(ev);

    auto do_compute = [this, context, file_name_string, ev,
         name_string, done] () {
      BundleReader reader(Env::Default(), file_name_string);
      auto s = reader.status();
      if (!s.ok()) {
        LOG(FATAL) << "Restore EV failure, create BundleReader error:"
                   << s.ToString();
      }

      EVRestoreDynamically(
          ev, name_string, partition_id_, partition_num_, context, &reader,
          "-partition_offset", "-keys", "-values", "-versions", "-freqs",
          reset_version_);
      ev->SetInitialized();
      done();
    };

    if (ev_async_restore_) {
      auto tp = KvRestoreThreadPool::GetInstance();
      tp->Schedule(do_compute);
    } else {
      do_compute();
    }
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  DataType counter_type_;
  int64 max_element_size_;
  float false_positive_probability_;
  TensorShape shape_;
  int64 steps_to_live_;
  bool restore_versions_;
  string ht_type_;
  int64 ht_partition_num_;
  int64 emb_index_;
  int64 slot_index_;
  int64 block_num_;
  int64 slot_num_;
  int64 filter_freq_;
  float l2_weight_threshold_;
  std::string layout_;
  int64 max_freq_;
  embedding::StorageType storage_type_;
  std::string storage_path_;
  std::vector<int64> storage_size_;
  int64 default_value_dim_;
  float default_value_no_permission_;
  bool record_freq_;
  bool record_version_;
  bool reset_version_;
  bool ev_async_restore_;
};

#define REGISTER_KERNELS(dev, ktype, vtype)                    \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV2")           \
                            .Device(DEVICE_##dev)              \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV2Op<ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)                        \
  REGISTER_KERNELS(dev, int32, type)                           \
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

template <typename Device, typename TKey, typename TValue>
class KvResourceImportV3Op: public AsyncOpKernel {
 public:
  explicit KvResourceImportV3Op(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("KvVariable dimension must be 1"));
    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
                 errors::InvalidArgument(
                    "partition_id must >= 0, ",
                    std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
                 errors::InvalidArgument(
                    "partition_num must >= 1, ",
                    std::to_string(partition_num_)));
    OP_REQUIRES_OK(c, c->GetAttr("reset_version", &reset_version_));
    bool reset_version = false;
    TF_CHECK_OK(ReadBoolFromEnvVar(
      "TF_EV_RESET_VERSION", false, &reset_version));
    reset_version_ = reset_version_ || reset_version;

    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EV_ASYNC_RESTORE", true,
                                   &ev_async_restore_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(2);
    const std::string name_string = name.scalar<string>()();

    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 1), &ev));

    core::ScopedUnref unref_me(ev);

    auto do_compute = [this, context, file_name_string, ev,
         name_string, done] () {
      BundleReader reader(Env::Default(), file_name_string);
      auto s = reader.status();
      if (!s.ok()) {
        LOG(FATAL) << "Restore EV failure, create BundleReader error:"
                   << s.ToString();
      }

      std::string name_string_temp(name_string);
      std::string new_str = "_";
      int64 pos = name_string_temp.find("/");
      while (pos != std::string::npos) {
        name_string_temp.replace(pos, 1, new_str.data(), 1);
        pos = name_string_temp.find("/");
      }
      std::string ssd_record_file_name =
          file_name_string + "-" + name_string_temp + "-ssd_record";
      //TODO: support change the partition number
      if (Env::Default()->FileExists(ssd_record_file_name + ".index").ok()) {
        std::string ssd_emb_file_name =
            file_name_string + "-" + name_string_temp + "-emb_files";
        if (ev->IsUsePersistentStorage()) {
          RestoreSsdRecord(ev, ssd_record_file_name, ssd_emb_file_name);
        } else {
          LoadSsdData(ev, ssd_record_file_name, ssd_emb_file_name);
        }
      }
      const Device& device = context->eigen_device<Device>();
      EVRestoreDynamically(
          ev, name_string, partition_id_, partition_num_, context, &reader,
          "-partition_offset", "-keys", "-values", "-versions", "-freqs",
          reset_version_, (Eigen::GpuDevice*)(&device));
      ev->SetInitialized();
      done();
    };

    if (ev_async_restore_) {
      auto tp = KvRestoreThreadPool::GetInstance();
      tp->Schedule(do_compute);
    } else {
      do_compute();
    }
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  TensorShape shape_;
  bool reset_version_;
  bool ev_async_restore_;
};

#define REGISTER_KERNELS(dev, ktype, vtype, device)            \
  REGISTER_KERNEL_BUILDER(Name("KvResourceImportV3")           \
                            .Device(DEVICE_##dev)              \
                            .HostMemory("prefix")              \
                            .HostMemory("tensor_names")        \
                            .HostMemory("empty_key")           \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceImportV3Op<device, ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type, device)                        \
  REGISTER_KERNELS(dev, int32, type, device)                           \
  REGISTER_KERNELS(dev, int64, type, device)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(CPU, type, CPUDevice)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#if GOOGLE_CUDA
#define REGISTER_KERNELS_GPU(type) REGISTER_KERNELS_ALL(GPU, type, GPUDevice)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS_GPU)
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class KvResourceIncrImportOp: public AsyncOpKernel {
 public:
  explicit KvResourceIncrImportOp(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));

    OP_REQUIRES_OK(c, c->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES(c, partition_id_ >= 0,
        errors::InvalidArgument(
          "partition_id must >= 0, ", std::to_string(partition_id_)));
    OP_REQUIRES_OK(c, c->GetAttr("partition_num", &partition_num_));
    OP_REQUIRES(c, partition_num_ >= 1,
        errors::InvalidArgument(
          "partition_num must >= 1, ", std::to_string(partition_num_)));

  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& file_name = context->input(0);
    const std::string file_name_string = file_name.scalar<string>()();
    const Tensor& name = context->input(2);
    const std::string name_string = name.scalar<string>()();

    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(context,
        LookupResource(context, HandleFromInput(context, 1), &ev));

    core::ScopedUnref unref_me(ev);

    BundleReader reader(Env::Default(), file_name_string);
    OP_REQUIRES_OK(context, reader.status());

    LOG(INFO) << "incr import, evname:"
              << name_string
              << "partition_num:"
              << partition_num_;

    EVRestoreDynamically(
        ev, name_string, partition_id_, partition_num_, context, &reader,
        "-incr_partition_offset", "-sparse_incr_keys", "-sparse_incr_values",
        "-sparse_incr_versions", "-sparse_incr_freqs");
    ev->SetInitialized();
    done();
  }

 private:
  int64 partition_id_;
  int64 partition_num_;
  DataType dtype_;
  TensorShape shape_;
  int64 steps_to_live_;
  bool restore_versions_;
  string ht_type_;
  int64 ht_partition_num_;
};


#define REGISTER_KERNELS(dev, ktype, vtype)                    \
  REGISTER_KERNEL_BUILDER(Name("KvResourceIncrImport")         \
                            .Device(DEVICE_##dev)              \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("dtype"),   \
                          KvResourceIncrImportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)                        \
  REGISTER_KERNELS(dev, int32, type)                           \
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

// Op that outputs tensors of all keys and all values.
template<typename TKey, typename TValue>
class KvResourceExportOp : public OpKernel {
 public:
  explicit KvResourceExportOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    EmbeddingVar<TKey, TValue> *ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    core::ScopedUnref unref_me(ev);
    std::vector<TKey> tot_key_list;
    std::vector<TValue *> tot_valueptr_list;
    std::vector<int64> tot_version_list;
    std::vector<int64> tot_freq_list;
    embedding::Iterator* it = nullptr;
    int64 total_size = ev->GetSnapshot(
        &tot_key_list, &tot_valueptr_list, &tot_version_list,
        &tot_freq_list, &it);

    // Create an output tensor
    Tensor *keys_output_tensor = NULL;
    Tensor *values_output_tensor = NULL;
    Tensor *versions_output_tensor = NULL;
    Tensor *freq_output_tensor = NULL;

    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          0, TensorShape({total_size}), &keys_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          1, TensorShape({total_size, ev->ValueLen()}),
          &values_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          2, TensorShape({tot_version_list.size()}),
          &versions_output_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
          3, TensorShape({tot_freq_list.size()}),
          &freq_output_tensor));

    auto keys_output = keys_output_tensor->template flat<TKey>();
    auto val_matrix = values_output_tensor->matrix<TValue>();
    auto versions_output = versions_output_tensor->template flat<int64>();
    auto freq_output = freq_output_tensor->template flat<int64>();

    for(size_t i = 0; i < total_size; i++) {
      keys_output(i) = tot_key_list[i];
      TValue *value = tot_valueptr_list[i];
      for(int64 m = 0; m < ev->ValueLen(); m++) {
        val_matrix(i, m) = *(value + m);
      }
      if (tot_version_list.size() != 0) {
        versions_output(i) = tot_version_list[i];
      }
      if (tot_freq_list.size() != 0) {
        freq_output(i) = tot_freq_list[i];
      }
    }
  }
};

#define REGISTER_KERNELS(dev, ktype, vtype)                    \
  REGISTER_KERNEL_BUILDER(Name("KvResourceExport")             \
                            .Device(DEVICE_##dev)              \
                            .TypeConstraint<ktype>("Tkeys")    \
                            .TypeConstraint<vtype>("Tvalues"), \
                          KvResourceExportOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL(dev, type)                        \
  REGISTER_KERNELS(dev, int32, type)                           \
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
}  // namespace tensorflow


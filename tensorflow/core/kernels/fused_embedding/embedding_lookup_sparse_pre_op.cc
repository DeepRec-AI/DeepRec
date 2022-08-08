#define EIGEN_USE_THREADS

#include <limits.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/bounds_check.h"

namespace tensorflow {

namespace{

struct IndicePair {
  int64_t row;
  int64_t column;
};

enum Part_Strategy {
  MOD,
  DIV
};

typedef void (*PARTITIONALGO)(
                        const int64_t* id_table, const int64_t numPartitions,
                        const int64_t idsPerPartition, const int64_t extras,
                        const int64_t originId, int64_t* segment, int64_t* newId);

template <Part_Strategy PS>
inline void GetPartitionIndex(
                        const int64_t* id_table, const int64_t numPartitions,
                        const int64_t idsPerPartition, const int64_t extras,
                        const int64_t originId, int64_t* segment, int64_t* newId) {}

template <>
inline void GetPartitionIndex<Part_Strategy::MOD>(
                        const int64_t* id_table, const int64_t numPartitions,
                        const int64_t idsPerPartition, const int64_t extras,
                        const int64_t originId, int64_t* segment, int64_t* newId) {
  *segment = originId % numPartitions;
  *newId = originId / numPartitions;
}

template <>
inline void GetPartitionIndex<Part_Strategy::DIV>(
                        const int64_t* id_table, const int64_t numPartitions,
                        const int64_t idsPerPartition, const int64_t extras,
                        const int64_t originId, int64_t* segment, int64_t* newId) {
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
  const int64_t *prange = id_table + numPartitions % 8;
  __m512i voffset = _mm512_set1_epi64(originId);
  int vectorSize = numPartitions / 8;
  for (int i = vectorSize - 1; i >= 0; --i) {
    __m512i vrange = _mm512_maskz_loadu_epi64(0xff, prange + i * 8);
    __mmask8 mask = _mm512_cmple_epi64_mask(vrange, voffset);
    if (mask != 0) {
      int numGreater = __builtin_ctz(mask);
      *segment = (numPartitions - 1) - 8 * (vectorSize - 1 - i) - numGreater;
      *newId = originId - id_table[*segment];
      return;
    }
  }

  for (int j = numPartitions % 8 - 1; j > -1; --j) {
    if (originId >= id_table[j]) {
      *segment = j;
      *newId = originId - id_table[j];
      break;
    }
  }
#else
  *segment = originId < extras * (idsPerPartition + 1) ?
            originId / (idsPerPartition + 1) :
            (originId - extras) / idsPerPartition;
  *newId = *segment < extras ?
            originId % (idsPerPartition + 1) :
            (originId - extras) % idsPerPartition;
#endif
}
}

typedef Eigen::ThreadPoolDevice CPUDevice;

class FusedEmbeddingSparsePreLookUpCPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePreLookUpCPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row", &fill_empty_row_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prune_invalid_id", &prune_invalid_id_));

    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_strategy", &partition_strategy_str_));
    if (partition_strategy_str_ == "div") {
      partition_strategy_ = GetPartitionIndex<Part_Strategy::DIV>;
    } else if (partition_strategy_str_ == "mod") {
      partition_strategy_ = GetPartitionIndex<Part_Strategy::MOD>;
    } else {
      OP_REQUIRES(ctx, false,
        errors::InvalidArgument("Not support partition_strategy type. ", partition_strategy_));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const int64_t default_id = default_id_ >= 0 ? default_id_ : 0;
    // 1. get input tensor
    Tensor const* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &values_tensor));
    const int64_t nnz = values_tensor->shape().dim_size(0);

    const int64_t* values = reinterpret_cast<const int64_t*>(
                                  values_tensor->flat<int64>().data());

    Tensor const* indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_indices", &indices_tensor));

    const int64_t* indices = reinterpret_cast<const int64_t*>(
                                  indices_tensor->flat<int64>().data());

    Tensor const* dense_shape = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &dense_shape));
    const int64_t batch_size = dense_shape->flat<int64>().data()[0];

    OpInputList partition_shapes;
    OP_REQUIRES_OK(ctx, ctx->input_list("partition_shapes", &partition_shapes));

    partition_total_sizes_ = 0;
    for (const Tensor& shape : partition_shapes) {
      OP_REQUIRES(ctx, shape.dims() <= 2,
                  errors::InvalidArgument(
                      "input partition_shapes must all less than rank 2"));
      partition_total_sizes_ += shape.flat<int64>().data()[0];
    }

    // fixme(marvin): show error info when got fake input.
    OP_REQUIRES(ctx, partition_total_sizes_ != 1,
        errors::InvalidArgument("Not support EV yet"));

    // 1.1 define output tensors
    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", &partitioned_values));
    OpOutputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->output_list("partitioned_indices", &partitioned_indices));

    Tensor* all_flags;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2 * num_partitions_,
                                  TensorShape{batch_size + nnz}, &all_flags));
    int32_t* all_flags_list = all_flags->flat<int32_t>().data();

    memset(all_flags_list, 0, (batch_size + nnz) * sizeof(int32_t));

    // 2.1 get index
    const int64_t idsPerPartition = partition_total_sizes_ / num_partitions_;
    const int64_t extras = partition_total_sizes_ % num_partitions_;
    std::vector<int64_t> empty_index_;
    // [p_seg_nums + list(p_seg, p_id)]
    int64_t* const id_index_array = new int64_t[num_partitions_ + 1 + nnz * 2];
    memset(id_index_array, 0, (num_partitions_ + 1) * sizeof(int64_t));

    // 2.2 get the map of the mutli-table index
    int64_t default_p_seg = 0;
    int64_t default_p_val = 0;
    int64_t p_seg = 0;
    int64_t p_val = 0;
    register int64_t tmp_id;
    int64_t* const min_id_per_seg = new int64_t[num_partitions_];
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
    int64_t* tmp_value_arr;

    // 2.1 build min_id_per_seg
    memset(min_id_per_seg, 0, (num_partitions_) * sizeof(int64_t));
    for (int i = 0; i < num_partitions_; ++i) {
      min_id_per_seg[i] = i < extras ?
      i * (idsPerPartition + 1) :
      i * idsPerPartition + extras;
    }

    //2.2.1 get new seg & id in id_index_array
    int64_t* new_p_seg;
    int64_t* new_p_id;
    int64_t* id_indices = id_index_array + num_partitions_ + 1;

    for (int64_t index = 0; index < nnz; ++index) {
      new_p_seg = id_indices + index * 2;
      new_p_id = id_indices + index * 2 + 1;

      // set default values;
      *(new_p_seg) = prune_invalid_id_ ? num_partitions_ : 0;
      *(new_p_id) = *(values + index);

      // set all_flags_list;
      all_flags_list[batch_size + index] = (*new_p_id < 0) ? 0 : 1;
      all_flags_list[*(indices + index * 2)] += !prune_invalid_id_ || !(*new_p_id < 0);

      partition_strategy_(min_id_per_seg, num_partitions_, idsPerPartition,
                          extras, *(new_p_seg + 1), new_p_seg, new_p_id);
      ++id_index_array[*new_p_seg];
    }

#else
    for (int64_t index = 0; index < nnz; ++index) {
      tmp_id = values[index];
      if (tmp_id < 0) {
        p_seg = prune_invalid_id_ ? num_partitions_ : 0;
        p_val = values[index];
        all_flags_list[*(indices + 2 * index)] += !p_seg;
      } else {
        all_flags_list[batch_size + index] = 1;
        ++all_flags_list[*(indices + 2 * index)];
        partition_strategy_(nullptr, num_partitions_, idsPerPartition,
                            extras, tmp_id, &p_seg, &p_val);
      }
      ++id_index_array[p_seg];
      *(id_index_array + 2 * index + num_partitions_ + 1) = p_seg;
      *(id_index_array + 2 * index + num_partitions_ + 2) = p_val;
    }
#endif

    // 2.3 fill_empty_row_index_
    if (fill_empty_row_) {
      // get default id p_seg_ and p_val_
      partition_strategy_(min_id_per_seg, num_partitions_, idsPerPartition, extras,
                          default_id, &default_p_seg, &default_p_val);
      for (int64_t origin_index = 0; origin_index < batch_size; ++origin_index) {
        if (all_flags_list[origin_index]) {
          all_flags_list[origin_index] = 0;
          continue;
        }
        all_flags_list[origin_index] = 1;
        empty_index_.push_back(origin_index);
        empty_index_.push_back(0);
      }
    }

    // 3 packaging the output tensor
    for (int i = 0; i < num_partitions_; ++i) {
      int64_t size = id_index_array[i];
      if (fill_empty_row_ && i == default_p_seg) {
        size += empty_index_.size() >> 1;
      }

      Tensor* sub_partitioned_values;
      OP_REQUIRES_OK(ctx, partitioned_values.allocate(
                              i, TensorShape({static_cast<int64_t>(size)}),
                              &sub_partitioned_values));
      int64_t* sub_p_values = reinterpret_cast<int64_t*>(
          sub_partitioned_values->flat<int64>().data());

      Tensor* sub_partitioned_indices;
      OP_REQUIRES_OK(ctx, partitioned_indices.allocate(
                              i, TensorShape({static_cast<int64_t>(size), 2}),
                              &sub_partitioned_indices));

      int64_t* sub_p_indces = reinterpret_cast<int64_t*>(
                                  sub_partitioned_indices->flat<int64>().data());
      if (!size) continue;

      int sub_part_index = 0;
      for (int index = 0; index < nnz; ++index) {
        if (id_index_array[(index) * 2 + num_partitions_ + 1] == i) {
          sub_p_values[sub_part_index] = id_index_array[(index) * 2 + num_partitions_ + 2];
          sub_p_indces[sub_part_index * 2] = *(indices + (index) * 2);
          sub_p_indces[sub_part_index * 2 + 1] = *(indices + (index) * 2 + 1);
          ++sub_part_index;
        }
      }
      if (fill_empty_row_ && default_p_seg == i) {
        memcpy(sub_p_indces + sub_part_index * 2,
          empty_index_.data(), empty_index_.size() * sizeof(int64_t));

        std::fill(sub_p_values + sub_part_index,
          sub_p_values + size, default_p_val);
      }
    }
    delete[] min_id_per_seg;
    delete[] id_index_array;
  }

 private:
  int num_partitions_;
  int partition_total_sizes_;
  int partition_axis_;
  bool fill_empty_row_;
  bool prune_invalid_id_;
  int64_t default_id_;
  PARTITIONALGO partition_strategy_;
  std::string partition_strategy_str_;
};

REGISTER_KERNEL_BUILDER(                  \
    Name("FusedEmbeddingSparsePreLookUp") \
    .Device(DEVICE_CPU)                   \
    .HostMemory("partition_shapes")       \
    .HostMemory("sp_dense_shape"),        \
    FusedEmbeddingSparsePreLookUpCPU);
}  // namespace tensorflow

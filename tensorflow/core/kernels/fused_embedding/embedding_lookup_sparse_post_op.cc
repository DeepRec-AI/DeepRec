#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/bounds_check.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

enum SparseSegmentReductionOperation { kSum, kMean, kSqrtN };

namespace {
    // input: input tensor value (it sores the id)
    // cols: How many elements to do SparseSegmentSum
    // output: rows * embedding_size
    template<typename T>
    static void sparse_gather_v1(T *input, int rows, int cols,
                                float *embedding_table, float *output,
                                int embedding_size, bool is_mean) {
      T *pidx = input;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < embedding_size; ++j) {
          float value = 0;
          int dense_num = 0;
          for (int k = 0; k < cols; ++k) {
            int embedding_row = (int)pidx[k];
            if (embedding_row >= 0) {
              value += embedding_table[embedding_row * embedding_size + j];
              dense_num += 1;
            }
          }

          if (is_mean && dense_num > 0) {
              *output++ = value / dense_num;
          } else {
              *output++ = value;
          }
        }
        pidx += cols;
      }
    }

    // embedding_size = 1
    template<typename T>
    static void sparse_gather_embeddingsize1(T *input, int rows, int cols,
                                            float *embedding_table, float *output,
                                            bool is_mean) {
    T *pidx = input;
    for (int i = 0; i < rows; ++i) {
      float value = 0;
      int dense_num = 0;
      for (int k = 0; k < cols; ++k) {
        int embedding_row = pidx[k];
        if (embedding_row >= 0) {
            value += embedding_table[embedding_row];
            dense_num += 1;
        }
      }
      if (is_mean && dense_num > 0) {
        *output++ = value / dense_num;
      } else {
        *output++ = value;
      }
      pidx += cols;
    }
    }

    // input cols = 1
    template<typename T>
    static void sparse_gather_column1(T *input, int rows, float *embedding_table,
                            float *output, int embedding_size) {
    T *pidx = input;
    for (int i = 0; i < rows; ++i) {
        int embedding_row = *pidx++;
        if (embedding_row >= 0) {
        float *pembedding = &embedding_table[embedding_row * embedding_size];
        for (int j = 0; j < embedding_size; ++j) {
            output[j] = pembedding[j];
        }
        } else {
        for (int j = 0; j < embedding_size; ++j) {
            output[j] = 0;
        }
        }
        output += embedding_size;
    }
    }

    template<typename T>
    static void sparse_gather(T *input, int rows, int cols, float *embedding_table,
                            float *output, int embedding_size, bool is_mean) {
      if (embedding_size == 1) {
        sparse_gather_embeddingsize1(input, rows, cols, embedding_table, output,
                                      is_mean);
      } else if (cols == 1) {
        sparse_gather_column1(input, rows, embedding_table, output, embedding_size);
      } else {
        //printf("General sparse gather!\n");
        sparse_gather_v1(input, rows, cols, embedding_table, output, embedding_size,
                        is_mean);
      }
    }

    // Use memcpy or manually assign?
    static void mycopy(float *dst, float *src, int float_num) {
      memcpy(dst, src, float_num * sizeof(float));
    }

    static void myadd(float *dst, float *src, int float_num) {
      for (int i = 0; i < float_num; ++i) {
        dst[i] += src[i];
      }
    }

    template<typename T>
    static void row_add(std::map<T *, std::vector<const T *>> &mapSet, int64 row_nums) {

      for (auto it = mapSet.begin(); it != mapSet.end(); ++it){
        T * dst = it->first;
        std::vector<const T *> srcs(std::move(it->second));
        int64 src_size = srcs.size();

        for (int row = 0; row < row_nums; ++row){
          dst[row] = 0.0;
          for (int index = 0; index < src_size; ++index) {
            dst[row] += srcs[index][row];
          }
        }
      }
    }

    template<typename T>
    static void row_add_mean(std::map<T *, std::vector<const T *>> &mapSet, int64 row_nums, bool is_mean) {

#define L(n) srcs[index + n][row]

      for (auto it = mapSet.begin(); it != mapSet.end(); ++it){
        T * dst = it->first;
        std::vector<const T *> srcs(std::move(it->second));
        int64 src_size = srcs.size();

        if (src_size==1){
          for (int row = 0; row < row_nums; ++row){
            dst[row] = srcs[0][row];
          }
          continue;
        }

        float sum_tmp = 0.0;
        int64 index = 0;
        int64 r = (src_size) % 8;
        int64 m = 1;
        if (src_size < 10 && is_mean) m = src_size;

        for (int row = 0; row < row_nums; ++row){
          sum_tmp = 0.0;
          index = 0;
          dst[row] = 0.0;
          switch (r) {
            case 2: {
              sum_tmp = (L(0) + L(1)) / m;
              dst[row] = sum_tmp;
              break;
            }
            case 3: {
              sum_tmp = (L(0) + L(1) + L(2)) / m;
              dst[row] = sum_tmp;
              break;
            }
            case 4: {
              sum_tmp = (L(0) + L(1) + L(2) + L(3)) / m;
              dst[row] = sum_tmp;
              break;
            }
            case 5: {
              sum_tmp = (L(0) + L(1) + L(2) + L(3) + L(4)) / m;
              dst[row] = sum_tmp;
              break;
            }
            case 6: {
              sum_tmp = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5)) / m;
              dst[row] = sum_tmp;
              break;
            }
            case 7: {
              sum_tmp = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6)) / m;
              dst[row] = sum_tmp;
              break;
            }
            case 0: {
              dst[row] = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7)) / m;
              index += 8;
              break;
            }
            case 1: {
              dst[row] = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7) + L(8)) / m;
              index += 8;
              break;
            }
          }
          for (index += r; index < src_size; index += 8) {
            sum_tmp = L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7);
            dst[row] += sum_tmp;
          }
          if (src_size >= 10 && is_mean) dst[row] /= src_size;
        }
      }
    }

    static void myscale(float *dst, float factor, int float_num) {
      for (int i = 0; i < float_num; ++i) {
          dst[i] *= factor;
      }
    }

    static void sparse_gather(int64 *input, int64 input_size, int64 *indice,
                            int indice_dim, int64 *shape, int rows, int cols,
                            float *embedding_table, float *output,
                            int embedding_size, bool is_mean) {
      // Record how many values in each row
      int *row_values = new int[rows];
      memset(row_values, 0, rows * sizeof(int));

      std::map<float *, std::vector<const float *>> mapSet;

      for (int64 i = 0; i < input_size; ++i) {
        int64 id = input[i];
        if (i < input_size && input[i] < 0) { // Skip invalid id
          continue;
        }
        auto row = indice[i * indice_dim];
        // for (int k = 1; k < indice_dim - 1; ++k) {
        //   row = row * shape[k] + indice[i * indice_dim + k];
        // }
        row_values[row] += 1;

        auto index = row * embedding_size;
        if (!mapSet.count(&output[index])){
          std::vector<const float *> srcs;
          mapSet[&output[index]] = srcs;
        }
        mapSet[&output[index]].push_back(&embedding_table[id * embedding_size]);
      }

      // row_add(mapSet, embedding_size);
      row_add_mean(mapSet, embedding_size, is_mean);

      for (int i = 0; i < rows; ++i) {
        if (row_values[i] == 0) {
          memset(&output[i * embedding_size], 0, embedding_size * sizeof(float));
        // } else if (is_mean && row_values[i] > 1) {
        //   float factor = 1.0f / row_values[i];
        //   myscale(&output[i * embedding_size], factor, embedding_size);
        }
      }
      delete[] row_values;
    }

    int64 partitioned_indices(std::vector<std::tuple<size_t, const int64*>> indices, int indice_dim, int64 id) {
        int indices_num = indices.size();
        int64 rows = 0;
        for (int i = 0; i < indices_num; ++i) {
            rows += std::get<0>(indices[i]);
            if (rows > id) {
                int idx = id - (rows - std::get<0>(indices[i]));
                return std::get<1>(indices[i])[idx * indice_dim];
            }
        }
    }

    const float* partitioned_embedding_tables(std::vector<std::tuple<size_t, const float*>> embedding_tables, int embedding_size, int64 id) {
        int tables_num = embedding_tables.size();
        int64 rows = 0;
        for (int i = 0; i < tables_num; ++i) {
            rows += std::get<0>(embedding_tables[i]);
            if (rows > id) {
                int idx = id - (rows - std::get<0>(embedding_tables[i]));
                return &(std::get<1>(embedding_tables[i])[idx * embedding_size]);
            }
        }
    }

    static void sparse_partitioned_gather(int64 input_size, std::vector<std::tuple<size_t, const int64*>> &indices,
                            int indice_dim, int rows,
                            std::vector<std::tuple<size_t, const float*>> &embedding_tables, float *output,
                            const int64_t embedding_size, SparseSegmentReductionOperation operation,
                            const bool set_empty_row_zero, const int *empty_row) {
      // Record how many values in each row
      int *row_values = new int[rows];
      memset(row_values, 0, rows * sizeof(int));

      std::map<float *, std::vector<const float *>> mapSet;

      // sum
      for (int64 i = 0; i < input_size; ++i) {
        int64 id = i;
        auto row = partitioned_indices(indices, indice_dim, i);
        row_values[row] += 1;

        auto index = row * embedding_size;
        if (!mapSet.count(&output[index])){
          std::vector<const float *> srcs;
          mapSet[&output[index]] = srcs;
        }
        mapSet[&output[index]].push_back(partitioned_embedding_tables(embedding_tables, embedding_size, id));
      }

      row_add_mean(mapSet, embedding_size, operation == SparseSegmentReductionOperation::kMean);

      for (int i = 0; i < rows; ++i) {
        // zero emtpy rows
        if (set_empty_row_zero && empty_row[i] == 1) {
            memset(&output[i * embedding_size], 0, embedding_size * sizeof(float));;
        }

        if (row_values[i] == 0) {
          memset(&output[i * embedding_size], 0, embedding_size * sizeof(float));
        } else if (operation == SparseSegmentReductionOperation::kMean && row_values[i] > 1) {
          // float factor = 1.0f / row_values[i];
          // myscale(&output[i * embedding_size], factor, embedding_size);
        } else if (operation == SparseSegmentReductionOperation::kSqrtN && row_values[i] > 1) {
          //fixme(marvin): shall we need care the data precision like "mean"? 
          float factor = 1.0f / std::sqrt(row_values[i]);
          myscale(&output[i * embedding_size], factor, embedding_size);
        }
      }
      delete[] row_values;
    }

    static void set_feature_nums(int32 *feature_nums, int64 input_size, std::vector<std::tuple<size_t, const int64*>> indices, int indice_dim) {
        for (int64 i = 0; i < input_size; ++i) {
            feature_nums[partitioned_indices(indices, indice_dim, i)]++;
        }
    }
}



template <typename Device>
class FusedSafeEmbeddingPostLookupOp : public OpKernel {
public:
  explicit FusedSafeEmbeddingPostLookupOp(OpKernelConstruction* ctx)
           : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
    if (combiner_ == "sum") {
      operation_ = SparseSegmentReductionOperation::kSum;
    } else if (combiner_ == "mean") {
      operation_ = SparseSegmentReductionOperation::kMean;
    } else if (combiner_ == "sqrtn") {
      operation_ = SparseSegmentReductionOperation::kSqrtN;
    } else {
      OP_REQUIRES(ctx, false,
          errors::InvalidArgument("Currently, 'mean', 'sqrtn' and 'sum' are only supported"));
    }
  }

  ~FusedSafeEmbeddingPostLookupOp() {}

  void Compute(OpKernelContext* ctx) override {
    OpInputList emb_shards;
    OP_REQUIRES_OK(ctx, ctx->input_list("emb_shards", &emb_shards));

    OpInputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->input_list("partitioned_indices", &partitioned_indices));

    Tensor const* dense_shape_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &dense_shape_tensor));

    Tensor const* row_empty_and_invalid_flags = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_empty_and_invalid_flags",
                                   &row_empty_and_invalid_flags));

    const int64_t embedding_size = emb_shards[0].shape().dim_size(1);

    int input_dims = dense_shape_tensor->dim_size(0);
    int batch_size = 1;
    for (int i = 0; i < input_dims - 1; ++i) {
      batch_size *= dense_shape_tensor->flat<int64>().data()[i];
    }

    // To check the input
    OP_REQUIRES(ctx, (dense_shape_tensor->dims() == 1),
                errors::InvalidArgument("Shape tensor is not valid (dims != 1)"));
    OP_REQUIRES(ctx, (dense_shape_tensor->dim_size(0) >= 2),
                errors::InvalidArgument("Shape tensor is not valid (dim_size(0) < 2)"));

    const int *empty_row = row_empty_and_invalid_flags->flat<int>().data();

    Tensor* emb_vectors_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size, embedding_size}),
                                  &emb_vectors_tensor));
    float *output = (float *)emb_vectors_tensor->tensor_data().data();

    Tensor* feature_nums_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({batch_size}), &feature_nums_tensor));
    int32 *feature_nums = (int32 *)feature_nums_tensor->tensor_data().data();
    memset(feature_nums, 0, batch_size*sizeof(int32));

    int64 input_size = 0;
    for (int i = 0; i < num_partitions_; ++i) {
      input_size += partitioned_indices[i].shape().dim_size(0);
    }

    int indice_dim = partitioned_indices[0].shape().dim_size(1);

    const bool set_empty_row_zero = default_id_ >= 0;

    std::vector<std::tuple<size_t, const float*>> embedding_tables;
    std::vector<std::tuple<size_t, const int64*>> indices;
    embedding_tables.reserve(num_partitions_);
    indices.reserve(num_partitions_);
    for (int i = 0; i < num_partitions_; i++) {
      const size_t sub_nnz = emb_shards[i].shape().dim_size(0);
      OP_REQUIRES(
          ctx, sub_nnz == partitioned_indices[i].shape().dim_size(0),
          errors::InvalidArgument(
              "emb_shard and partitioned_indice dosn't have the same length"));
      embedding_tables.emplace_back(std::make_tuple(sub_nnz, emb_shards[i].flat<float>().data()));
      indices.emplace_back(std::make_tuple(sub_nnz, partitioned_indices[i].flat<int64>().data()));
    }

    sparse_partitioned_gather(input_size, indices, indice_dim, batch_size,
            embedding_tables, output, embedding_size, operation_, set_empty_row_zero, empty_row);
    set_feature_nums(feature_nums, input_size, indices, indice_dim);
  }

private:
int num_partitions_;
  int partition_axis_;
  std::string combiner_;
  float max_norm_;
  int64_t default_id_;
  SparseSegmentReductionOperation operation_;
};

REGISTER_KERNEL_BUILDER(                                        \
    Name("FusedEmbeddingSparsePostLookUp")                      \
    .Device(DEVICE_CPU),                                        \
    FusedSafeEmbeddingPostLookupOp<CPUDevice>);


template <typename Device>
class FusedSafeEmbeddingPostLookupGradOp : public OpKernel {
 public:
  explicit FusedSafeEmbeddingPostLookupGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
    if (combiner_ == "sum") {
      operation_ = SparseSegmentReductionOperation::kSum;
    } else if (combiner_ == "mean") {
      operation_ = SparseSegmentReductionOperation::kMean;
    } else if (combiner_ == "sqrtn") {
      operation_ = SparseSegmentReductionOperation::kSqrtN;
    } else {
      OP_REQUIRES(ctx, false,
          errors::InvalidArgument("Currently, 'mean', 'sqrtn' and 'sum' are only supported"));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor const* top_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("top_grad", &top_grad_tensor));

    OpInputList emb_shards;
    OP_REQUIRES_OK(ctx, ctx->input_list("emb_shards", &emb_shards));

    OpInputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->input_list("partitioned_indices", &partitioned_indices));

    Tensor const* feature_nums = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("feature_nums", &feature_nums));

    Tensor const* row_empty_and_invalid_flags = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_empty_and_invalid_flags",
                                   &row_empty_and_invalid_flags));

    OpOutputList grad_shards;
    OP_REQUIRES_OK(ctx, ctx->output_list("grad_shards", &grad_shards));

    const float *top_grad = top_grad_tensor->flat<float>().data();
    const int64_t batch_size = top_grad_tensor->shape().dim_size(0);
    const int64_t emb_vec_size = emb_shards[0].shape().dim_size(1);
    const int *f_nums = feature_nums->flat<int>().data();
    const int *empty_row = row_empty_and_invalid_flags->flat<int>().data();

    const bool set_empty_row_zero = default_id_ >= 0;

    for (int i = 0; i < num_partitions_; i++) {
      const int64_t sub_nnz = partitioned_indices[i].shape().dim_size(0);
      const int64_t indices_col = partitioned_indices[i].shape().dim_size(1);
      const int64 *indices = partitioned_indices[i].flat<int64>().data();
      Tensor* grad_shard;
      OP_REQUIRES_OK(
          ctx, grad_shards.allocate(i, TensorShape({sub_nnz, emb_vec_size}),
                                    &grad_shard));
      float *grad = grad_shard->flat<float>().data();

      std::vector<float> l2_norm(sub_nnz, 1.0);
      if (max_norm_ > 0.0) {
        const float *emb = emb_shards[i].flat<float>().data();
        for (int j = 0; j < sub_nnz; ++j) {
          float sum = 0.0;
          for (int k = 0; k < emb_vec_size; ++k) {
            sum += emb[j * emb_vec_size + k] * emb[j * emb_vec_size + k];
          }
          l2_norm[j] = std::sqrt(sum);
        }
      }

      if (operation_ == SparseSegmentReductionOperation::kSum) {
        for (int j = 0 ; j < sub_nnz; ++j) {
          int64 idx = indices[j*indices_col];
          if (set_empty_row_zero == true && empty_row[idx] == 1)
            memset(&grad[j*emb_vec_size], 0, sizeof(float)*emb_vec_size);
          else
            memcpy(&grad[j*emb_vec_size], &top_grad[idx*emb_vec_size], sizeof(float)*emb_vec_size);
        }
      }
      else if (operation_ == SparseSegmentReductionOperation::kMean) {
        for (int j = 0 ; j < sub_nnz; ++j) {
          int64 idx = indices[j*indices_col];
          if (set_empty_row_zero == true && empty_row[idx] == 1)
            memset(&grad[j*emb_vec_size], 0, sizeof(float)*emb_vec_size);
          else {
            for (int k = 0; k < emb_vec_size; ++k) {
              grad[j*emb_vec_size + k] = top_grad[idx*emb_vec_size + k] / f_nums[idx];
              if (l2_norm[j] > max_norm_) {
                grad[j*emb_vec_size + k] *= max_norm_ / l2_norm[j];
              }
            }
          }
        }
      }
      else if (operation_ == SparseSegmentReductionOperation::kSqrtN) {
        for (int j = 0 ; j < sub_nnz; ++j) {
          int64 idx = indices[j*indices_col];
          if (set_empty_row_zero == true && empty_row[idx] == 1)
            memset(&grad[j*emb_vec_size], 0, sizeof(float)*emb_vec_size);
          else {
            for (int k = 0; k < emb_vec_size; ++k) {
              grad[j*emb_vec_size + k] = top_grad[idx*emb_vec_size + k] / std::sqrt(f_nums[idx]);
              if (l2_norm[j] > max_norm_) {
                grad[j*emb_vec_size + k] *= max_norm_ / l2_norm[j];
              }
            }
          }
        }
      }
      else {
        OP_REQUIRES(ctx, false,
          errors::InvalidArgument("Currently, 'mean', 'sqrtn' and 'sum' are only supported"));
      }
    }
  }

private:
  int num_partitions_;
  int partition_axis_;
  std::string combiner_;
  float max_norm_;
  int64_t default_id_;
  SparseSegmentReductionOperation operation_;
};

REGISTER_KERNEL_BUILDER(                                        \
    Name("FusedEmbeddingSparsePostLookUpGrad")                  \
    .Device(DEVICE_CPU),                                        \
    FusedSafeEmbeddingPostLookupGradOp<CPUDevice>);

}  // namespace tensorflow

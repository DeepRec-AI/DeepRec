#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/bounds_check.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

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
    static void row_add(std::map<T *, std::vector<T *>> &mapSet, int64 row_nums) {

      for (auto it = mapSet.begin(); it != mapSet.end(); ++it){
        T * dst = it->first;
        std::vector<T *> srcs(std::move(it->second));
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
    static void row_add_mean(std::map<T *, std::vector<T *>> &mapSet, int64 row_nums, bool is_mean) {

#define L(n) srcs[index + n][row]

      for (auto it = mapSet.begin(); it != mapSet.end(); ++it){
        T * dst = it->first;
        std::vector<T *> srcs(std::move(it->second));
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

    template<typename Tid, typename Tshape>
    static void sparse_gather(Tid *input, int64 input_size, Tshape *indice,
                            int indice_dim, Tshape *shape, int rows, int cols,
                            float *embedding_table, float *output,
                            int embedding_size, bool is_mean) {
      // Record how many values in each row
      int *row_values = new int[rows];
      memset(row_values, 0, rows * sizeof(int));

      std::map<float *, std::vector<float *>> mapSet;

      for (int64 i = 0; i < input_size; ++i) {
        Tid id = input[i];
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
          std::vector<float *> srcs;
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
}

/*
  sample: [['green' 'red' 'blue' 'yellow' 'pink' 'blue' 'red' 'indigo']
           ['' '' '' '' '' '' '' '']
           ['' '' '' 'yellow' 'pink' 'blue' 'red' 'indigo']
           ['' '' '' '' '' '' '' '']
           ['green' '' '' '' '' '' '' '']]
     =>   [[ True  True  True  True  True  True  True  True]
           [False False False False False False False False]
           [False False False  True  True  True  True  True]
           [False False False False False False False False]
           [ True False False False False False False False]]
--------------------------------------------------------------------------------------
  weight: float[[ 0.23860918  0.07992432 -0.7441818 ]
                [-0.8256738  -0.50271106  0.39016065]
                [-0.7978571   0.3993331  -0.12494776]
                [-0.555991   -0.6705441  -0.23192379]
                [-0.5283828   0.19715567  0.12184268]]
  input: int64[4 0 0 1 1 0 0 1 1 1 0 0 1 4] from StringToHashBucketFast output
  dense_shape: int64[5 8]
  indice: int64[[0 0] from to_sparse_input/indices(Where) output
                [0 1]
                [0 2]
                [0 3]
                [0 4]
                [0 5]
                [0 6]
                [0 7]
                [2 3]
                [2 4]
                [2 5]
                [2 6]
                [2 7]
                [4 0]]
    embedded: float[[-0.25637093 -0.12391002 -0.21055032]
                    [ 0.          0.          0.        ]
                    [-0.3999606  -0.2696569  -0.06357633]
                    [ 0.          0.          0.        ]
                    [-0.5283828   0.19715567  0.12184268]]
-----------------------------------------------------------------------------------
      input_size: sum of input tensor size == 14
      indice_dim: dim_size(1) of indice tensor[14, 2] == 2
      shape: dense_shape == [5 8]
      batch_size: dim of dense_shape == 5
      cols: dim_size(1) of dense_shape == 8
      embedding_size: dim_size(1) of weight tensor == 3
      sparse_gather(input, input_size, indice, indice_dim, shape, batch_size,
                    cols, weight, output, embedding_size, is_mean);
*/

template <typename Device, typename Tid, typename Tshape>
class FusedSafeEmbeddingLookupSparseLocalOp : public OpKernel {
public:
  explicit FusedSafeEmbeddingLookupSparseLocalOp(OpKernelConstruction* context)
           : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_));
    //OP_REQUIRES_OK(context, context->GetAttr("Dims", &dims));
    node_name = context->def().name();
  }

  ~FusedSafeEmbeddingLookupSparseLocalOp() {}

  void Compute(OpKernelContext* context) override {
    // Grab the weight
    float *weight;
    const Tensor* weight_tensor = &context->input(0);

    // for saved model
    if (weight_tensor->dtype() == DT_RESOURCE) {
      Var* variable;
      OP_REQUIRES_OK(context,
                     LookupResource(context, HandleFromInput(context, 0), 
                                    &variable));
      core::ScopedUnref s(variable);
      weight_tensor = variable->tensor();
      OP_REQUIRES(context, weight_tensor->dtype() == DT_FLOAT,
                  errors::InvalidArgument("Expect float weight in ",
                                          node_name));
    }

    weight = (float *)weight_tensor->tensor_data().data();
    
    // Input id
    const Tensor& input_tensor = context->input(1);
    Tid *input = (Tid *)input_tensor.tensor_data().data();
    
    const Tensor& shape_tensor = context->input(2);
    Tshape *shape = (Tshape *)shape_tensor.tensor_data().data();

    // To check the input
    OP_REQUIRES(context, (shape_tensor.dims() == 1),
                errors::InvalidArgument("Shape tensor is not valid (dims != 1)"));
    OP_REQUIRES(context, (shape_tensor.dim_size(0) >= 2),
                errors::InvalidArgument("Shape tensor is not valid (dim_size(0) < 2)"));

    int64 input_size = 1;
    for (int i = 0; i < input_tensor.dims(); ++i) {
      input_size *= input_tensor.dim_size(i);
    }
    
    int input_dims = shape_tensor.dim_size(0);
    int cols = shape[input_dims - 1];
    int batch_size = 1;
    for (int i = 0; i < input_dims - 1; ++i) {
      batch_size *= shape[i];
    }
    int embedding_size = weight_tensor->dim_size(1);
    bool is_mean = (combiner_ == "mean");

    const Tensor& indice_tensor = context->input(3);
    Tshape *indice = (Tshape *)indice_tensor.tensor_data().data();
    int indice_dim = indice_tensor.dim_size(1);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape output_shape({batch_size, embedding_size});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    float *output = (float *)output_tensor->tensor_data().data();

    if (false && input_size == batch_size * cols) { // input id is dense
      //fixme(marvin): disable this branch just for test.
      sparse_gather(input, batch_size, cols, weight, output, embedding_size, is_mean);
    } else { // input id is sparse
      OP_REQUIRES(context, (indice_tensor.dims() == 2),
                  errors::InvalidArgument("Indice tensor is not as expected (dims != 2)"));
      OP_REQUIRES(context, (indice_tensor.dim_size(0) == input_size),
                  errors::InvalidArgument("Indice tensor is not as expected (dim_size(0) != batch_size)"));
      sparse_gather(input, input_size, indice, indice_dim, shape, batch_size,
                    cols, weight, output, embedding_size, is_mean);
    }
  }

private:
  std::string combiner_;
  std::string node_name;
};

REGISTER_KERNEL_BUILDER(                                        \
    Name("FusedSafeEmbeddingLookupSparseLocal")                 \
    .Device(DEVICE_CPU)                                         \
    .TypeConstraint<int32>("T_id")                               \
    .TypeConstraint<int64>("T_shape"),                           \
    FusedSafeEmbeddingLookupSparseLocalOp<CPUDevice, int32, int64>);

REGISTER_KERNEL_BUILDER(                                        \
    Name("FusedSafeEmbeddingLookupSparseLocal")                 \
    .Device(DEVICE_CPU)                                         \
    .TypeConstraint<int64>("T_id")                               \
    .TypeConstraint<int64>("T_shape"),                           \
    FusedSafeEmbeddingLookupSparseLocalOp<CPUDevice, int64, int64>);


enum class SparseSegmentReductionOperation { kSum, kMean, kSqrtN };

namespace functor {

template <typename T, typename Index, typename SegmentId>
struct SparseSegmentGradFunctor {
  void operator()(OpKernelContext* context,
                  SparseSegmentReductionOperation operation,
                  typename TTypes<T>::ConstMatrix input_flat,
                  typename TTypes<Index>::ConstVec indices_vec,
                  typename TTypes<SegmentId>::ConstVec segment_vec,
                  typename TTypes<T>::Matrix output_flat) {
    const int64_t N = indices_vec.size();
    const SegmentId M = output_flat.dimension(0);

    // Note that similar to SparseSegmentMean, we assume that segment_vec is
    // already sorted and has non-negative values.
    const SegmentId num_segments = input_flat.dimension(0);
    const SegmentId last_segment_id_plus_one =
        internal::SubtleMustCopy(segment_vec(N - 1)) + 1;
    OP_REQUIRES(context, last_segment_id_plus_one <= num_segments,
                errors::InvalidArgument("Invalid number of segments"));

    // Compute scaling factors for input.
    std::vector<double> scaling(
        (operation == SparseSegmentReductionOperation::kSum ? 0 : num_segments),
        0.0);
    if (operation != SparseSegmentReductionOperation::kSum) {
      for (int64_t i = 0; i < N; ++i) {
        const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
        OP_REQUIRES(
            context, FastBoundsCheck(idx, num_segments),
            errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                    num_segments, ")."));
        scaling[idx] += 1;
      }
      for (size_t i = 0; i < scaling.size(); ++i) {
        switch (operation) {
          case SparseSegmentReductionOperation::kSum: {
            OP_REQUIRES(
                context, false,
                errors::Internal(
                    "Should not happen: sum inside SparseSegmentReductionOp "
                    "scaling generation."));
          }
          case SparseSegmentReductionOperation::kMean: {
            scaling[i] = 1.0 / std::max(scaling[i], 1.0);
            break;
          }
          case SparseSegmentReductionOperation::kSqrtN: {
            scaling[i] = 1.0 / sqrt(std::max(scaling[i], 1.0));
            break;
          }
            // No default to get compiler warnings for missing cases.
        }
      }
    }

    output_flat.setZero();
    std::vector<bool> is_modified(M, false);

    for (int64_t i = 0; i < N; ++i) {
      const Index output_idx = internal::SubtleMustCopy(indices_vec(i));
      OP_REQUIRES(context, FastBoundsCheck(output_idx, M),
                  errors::InvalidArgument("Index ", output_idx,
                                          " out of range [0, ", M, ")."));

      const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(idx, num_segments),
          errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                  num_segments, ")."));

      const T scale = (operation == SparseSegmentReductionOperation::kSum
                           ? static_cast<T>(1)
                           : static_cast<T>(scaling[idx]));
      if (is_modified[output_idx]) {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx) * scale;
        }
      } else {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx) * scale;
        }
      }
      is_modified[output_idx] = true;
    }
  }
};

}  // namespace functor

template <typename Device, typename T, typename Tinput, typename Tindices, typename Tdense_shape>
class FusedSafeEmbeddingLookupSparseLocalGradOp : public OpKernel {
public:
  explicit FusedSafeEmbeddingLookupSparseLocalGradOp(OpKernelConstruction* context)
           : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_));
    //OP_REQUIRES_OK(context, context->GetAttr("Dims", &dims));

    if (combiner_ == "sum") {
      operation_ = SparseSegmentReductionOperation::kSum;
    } else if (combiner_ == "mean") {
      operation_ = SparseSegmentReductionOperation::kMean;
    } else if (combiner_ == "sqrtn") {
      operation_ = SparseSegmentReductionOperation::kSqrtN;
    } else {
      OP_REQUIRES(context, false,
          errors::InvalidArgument("Currently, 'mean', 'sqrtn' and 'sum' are only supported"));
    }

    node_name = context->def().name();

    static bool printed = false;
    if (!printed) {
      printf("******** FusedSafeEmbeddingLookupSparseLocalGradOp ********\n");
      printed = true;
    }
  }

  ~FusedSafeEmbeddingLookupSparseLocalGradOp() {
  }

  void Compute(OpKernelContext* context) override {
    // Grab gradients
    const Tensor& gradients_tensor = context->input(0);
    T *gradients = (T *)gradients_tensor.tensor_data().data();
    OP_REQUIRES(context, (gradients_tensor.dims() == 2),
                errors::InvalidArgument("Gradients tensor is not valid (dims != 2)"));
    int64 gradients_row = gradients_tensor.dim_size(0);
    int64 embedding_col = gradients_tensor.dim_size(1);

    // Grad input hash value
    const Tensor& input_tensor = context->input(1);
    Tinput *input = (Tinput *)input_tensor.tensor_data().data();
    int64 input_size = 1;
    for (int i = 0; i < input_tensor.dims(); ++i) {
      input_size *= input_tensor.dim_size(i);
    }

    // Grad indices value
    const Tensor& indices_tensor = context->input(2);
    Tindices *indices_ptr = (Tindices *)indices_tensor.tensor_data().data();
    int indices_row = indices_tensor.dim_size(0);
    int indices_col = indices_tensor.dim_size(1);
    OP_REQUIRES(context, (indices_tensor.dims() == 2),
                errors::InvalidArgument("Indice tensor is not as expected (dims != 2)"));
    OP_REQUIRES(context, (indices_tensor.dim_size(0) == input_size),
                errors::InvalidArgument("Indice tensor is not as expected (dim_size(0) != batch_size)"));
    std::vector<Tindices> input_indices; // collect first col
    for (int64 i = 0; i < indices_row; ++i) {
      input_indices.emplace_back(indices_ptr[i*indices_col]);
    }

    // Grad input dense shape
    const Tensor& dense_shape_tensor = context->input(3);
    Tdense_shape *dense_shape = (Tdense_shape *)dense_shape_tensor.tensor_data().data();
    OP_REQUIRES(context, (dense_shape_tensor.dims() == 1),
                errors::InvalidArgument("Shape tensor is not valid (dims != 1)"));
    OP_REQUIRES(context, (dense_shape_tensor.dim_size(0) >= 2),
                errors::InvalidArgument("Shape tensor is not valid (dim_size(0) < 2)"));
    int input_dims = dense_shape_tensor.dim_size(0);
    int input_cols = dense_shape[input_dims - 1];
    int batch_size = 1;
    for (int i = 0; i < input_dims - 1; ++i) {
      batch_size *= dense_shape[i];
    }
    OP_REQUIRES(context, (gradients_row == batch_size),
                errors::InvalidArgument("gradients row is not same as batch_size)"));

    // Grad combiner
    // bool is_mean = (combiner == 1);

    // compute unique value and indices of input hash value
    std::vector<Tinput> unique_value;
    std::vector<Tinput> unique_indices;
    unique_value.reserve(input_size);
    unique_indices.reserve(input_size);
    for (int64 i = 0; i < input_size; ++i) {
        Tinput id = input[i];
        if (id < 0) { // Skip invalid id
          continue;
        }
        auto it = std::find(unique_value.begin(), unique_value.end(), id);
        if (it == unique_value.end()) { // no find
          unique_indices.push_back(unique_value.size());
          unique_value.push_back(id);
        }
        else {
          unique_indices.push_back(it - unique_value.begin());
        }
    }

    // printf("unique_indices: ");
    // for (int i = 0; i < unique_indices.size(); ++i)
    //   printf("%d ", unique_indices[i]);
    // printf("\n");

    // printf("input_indices: ");
    // for (int i = 0; i < input_indices.size(); ++i)
    //   printf("%d ", input_indices[i]);
    // printf("\n");

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape output_shape({unique_value.size(), embedding_col});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    output_tensor->flat<T>().setZero();
    T *output = (T *)output_tensor->tensor_data().data();
    
    memset(output, 0, embedding_col * sizeof(float) * unique_value.size());

    Tensor* unique_tensor = NULL;
    TensorShape unique_shape({unique_value.size()});
    OP_REQUIRES_OK(context, context->allocate_output(1, unique_shape, &unique_tensor));
    Tinput *unique = (Tinput *)unique_tensor->tensor_data().data();

    int64 unique_num = unique_value.size();
    for (int64 i = 0; i < unique_num; ++i) {
      unique[i] = unique_value[i];
    }

    // if (input_size == batch_size * input_cols) { // input id is dense
    // } else { // input id is sparse
    // }

    if (operation_ == SparseSegmentReductionOperation::kMean) {
      auto input_flat = gradients_tensor.flat_outer_dims<T>();
      typename TTypes<Tinput>::ConstVec indices_vec(unique_indices.data(), unique_indices.size());
      typename TTypes<Tindices>::ConstVec segment_vec(input_indices.data(), input_indices.size());
      auto output_flat = output_tensor->flat_outer_dims<T>();
      functor::SparseSegmentGradFunctor<T, Tinput, Tindices>()(
          context, operation_, input_flat, indices_vec, segment_vec, output_flat);
    }
    else if (operation_ == SparseSegmentReductionOperation::kSum) {
      uint64 rows = unique_indices.size();
      // std::vector<int64> row_values(unique_value.size(), 0);
      std::map<float *, std::vector<float *>> mapSet;

      for (int64 i = 0; i < rows; ++i) {
        // row_values[unique_indices[i]] += 1;

        auto index = unique_indices[i] * embedding_col;
        // memset(&output[index * embedding_col], 0, embedding_col * sizeof(float));
        if (!mapSet.count(&output[index])){
          std::vector<float *> srcs;
          mapSet[&output[index]] = srcs;
        }
        mapSet[&output[index]].push_back(&gradients[input_indices[i] * embedding_col]);

      }

      row_add(mapSet, embedding_col);
      // printf("******Goto row_add_mean func.******\n");
      // row_add_mean(mapSet, embedding_col, false);
      
      // for (int i = 0; i < unique_value.size(); ++i) {
      //   if (row_values[i] == 0) {
      //     memset(&output[i * embedding_col], 0, embedding_col * sizeof(float));
      //   }
      // }
      // delete[] row_values;

    }
    else if (operation_ == SparseSegmentReductionOperation::kSqrtN) {

    }
  }

private:
  template <typename Tdata>
  void copy(Tdata* dst, const Tdata* src, const int64 num) {
    memcpy(dst, src, num * sizeof(T));
  }

  template <typename Tdata>
  void add(Tdata* dst, const Tdata* src, const int64 num) {
    for (int64 i = 0; i < num; ++i) {
      dst[i] += src[i];
    }
  }

  template <typename Tdata>
  void scale(Tdata* dst, const Tdata factor, const int64 num) {
    for (int64 i = 0; i < num; ++i) {
      dst[i] *= factor;
    }
  }

private:
  std::string combiner_;
  std::string node_name;
  SparseSegmentReductionOperation operation_;
};

REGISTER_KERNEL_BUILDER(                            \
    Name("FusedSafeEmbeddingLookupSparseLocalGrad") \
    .Device(DEVICE_CPU)                             \
    .TypeConstraint<float>("T")                     \
    .TypeConstraint<int64>("Tinput")                \
    .TypeConstraint<int32>("Tindices")              \
    .TypeConstraint<int64>("Tdense_shape"),         \
    FusedSafeEmbeddingLookupSparseLocalGradOp<CPUDevice, float, int64, int32, int64>);

REGISTER_KERNEL_BUILDER(                            \
    Name("FusedSafeEmbeddingLookupSparseLocalGrad") \
    .Device(DEVICE_CPU)                             \
    .TypeConstraint<float>("T")                     \
    .TypeConstraint<int64>("Tinput")                \
    .TypeConstraint<int64>("Tindices")              \
    .TypeConstraint<int64>("Tdense_shape"),         \
    FusedSafeEmbeddingLookupSparseLocalGradOp<CPUDevice, float, int64, int64, int64>);

}  // namespace tensorflow

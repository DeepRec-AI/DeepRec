#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/sparse_fill_empty_rows_op_util.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
namespace functor {

template <typename T, typename Tindex>
__global__ void
parallel_sparse_fill_empty_rows_kernel(const Tindex *indices,
				       const Tindex *scratch,
				       const Tindex *offset,
				       const T *values,
				       const T *default_value,
				       const Tindex *empty_row_index,
				       const Tindex empty_row_index_size,
				       Tindex *output_indices,
				       T *output_values,
				       Tindex *reverse_index_map,
				       const Tindex N,
				       const int rank) {
  // Fill in values for rows that are not missing
  GPU_1D_KERNEL_LOOP(index, N*rank) {
    const Tindex dim_index = indices[index];
    const Tindex real_index = index / rank;
    const Tindex dim_offset = index % rank;
    const Tindex row = indices[index-dim_offset];
    const Tindex output_i = ((row == 0) ? 0 : scratch[row-1]) + offset[real_index];
    output_indices[output_i*rank + dim_offset] = dim_index;
    output_values[output_i] = values[real_index];
    reverse_index_map[real_index] = output_i;
  }
  
  // Fill in values for rows that are missing
  GPU_1D_KERNEL_LOOP(index, empty_row_index_size*rank) {
    const Tindex real_index = index / rank;
    const Tindex dim_offset = index % rank;
    const Tindex row = empty_row_index[real_index];
    const Tindex starting_index = (row == 0) ? 0 : scratch[row-1];
    output_indices[starting_index*rank + dim_offset] = (dim_offset == 0 ? row : 0);
    output_values[starting_index] = *default_value;
  }
}    
  
template <typename T, typename Tindex>
struct ParallelSparseFillEmptyRows<GPUDevice, T, Tindex> {
  void operator()(const GPUDevice& d,
		  OpKernelContext *context,
		  Allocator *allocator,
		  typename TTypes<Tindex>::ConstMatrix indices,
		  typename TTypes<T>::ConstVec values,
		  typename TTypes<T>::ConstScalar default_val,
		  typename TTypes<Tindex>::Vec reverse_index_map,
		  Tensor *output_indices_t,
		  Tensor *output_values_t,
		  std::vector<Tindex> &scratch,
		  const std::vector<Tindex> &offset,
		  const Tindex dense_rows,
		  const Tindex N,
		  TTypes<bool>::Vec empty_row_indicator,		    
		  const int rank) {
    std::vector<uint8_t> empty_row_indicator_vec(dense_rows, false);
    std::vector<Tindex> empty_row_index;
    for (Tindex row = 0; row < dense_rows; ++row) {
      // Scratch here describes the number of elements in this dense row
      empty_row_indicator_vec[row] = (scratch[row] == 0);
      if (empty_row_indicator_vec[row])
	empty_row_index.push_back(row);
      // In filled version, each row has at least one element.
      scratch[row] = std::max(scratch[row], Tindex{1});
      // Update scratch to represent the number of elements up to and
      // including dense_row + 1:
      //   scratch(0) == #{elements of row 0}
      //   scratch(1) == #{elements of row 1} + #{elements of row 0}
      // ...
      //   scratch(i) == starting index for elements in row i+1.
      if (row > 0)
	scratch[row] += scratch[row-1];
    }

    const Tindex N_full = scratch[dense_rows-1];
    TensorShape output_indices_shape({N_full, rank});
    OP_REQUIRES_OK(context,
		   context->allocate_temp(DataTypeToEnum<Tindex>::v(),
					  output_indices_shape,
					  output_indices_t));
    OP_REQUIRES_OK(context,
		   context->allocate_temp(DataTypeToEnum<T>::v(),
					  TensorShape({N_full}),
					  output_values_t));
    auto output_indices = output_indices_t->matrix<Tindex>();
    auto output_values = output_values_t->vec<T>();
    
    // allocator global memory for indices, scratch and offset
    Tindex *d_indices =
      TypedAllocator::Allocate<Tindex>(allocator, indices.size(), AllocationAttributes());
    Tindex *d_scratch =
      TypedAllocator::Allocate<Tindex>(allocator, scratch.size(), AllocationAttributes());
    Tindex *d_offset =
      TypedAllocator::Allocate<Tindex>(allocator, offset.size(), AllocationAttributes());
    Tindex* d_empty_row_index = TypedAllocator::Allocate<Tindex>(
        allocator, empty_row_index.size(), AllocationAttributes());
    
    // copy memory from host to device.
    cudaMemcpyAsync(d_indices, indices.data(), indices.size()*sizeof(Tindex), cudaMemcpyHostToDevice, d.stream());
    cudaMemcpyAsync(d_scratch, scratch.data(), scratch.size()*sizeof(Tindex), cudaMemcpyHostToDevice, d.stream());
    cudaMemcpyAsync(d_offset, offset.data(), offset.size()*sizeof(Tindex), cudaMemcpyHostToDevice, d.stream());
    cudaMemcpyAsync(d_empty_row_index, empty_row_index.data(),
                    empty_row_index.size() * sizeof(Tindex),
                    cudaMemcpyHostToDevice, d.stream());
    cudaStreamSynchronize(d.stream());

    const Tindex empty_row_index_size = empty_row_index.size();
    const int work_item = std::max(N*rank, empty_row_index_size);
    GpuLaunchConfig config = GetGpuLaunchConfig(work_item, d);
    GpuLaunchKernel(parallel_sparse_fill_empty_rows_kernel<T, Tindex>,
		    config.block_count, config.thread_per_block, 0, d.stream(),
		    d_indices, d_scratch, d_offset, values.data(),
		    default_val.data(), d_empty_row_index, empty_row_index_size,
		    output_indices.data(), output_values.data(),
		    reverse_index_map.data(), N, rank);

    // Fill in values for empty row indicator
    cudaMemcpyAsync(empty_row_indicator.data(), empty_row_indicator_vec.data(),
		    sizeof(bool)*dense_rows, cudaMemcpyHostToDevice, d.stream());
    
    TypedAllocator::Deallocate<Tindex>(allocator, d_indices, indices.size());
    TypedAllocator::Deallocate<Tindex>(allocator, d_scratch, scratch.size());
    TypedAllocator::Deallocate<Tindex>(allocator, d_offset, offset.size());
    TypedAllocator::Deallocate<Tindex>(allocator, d_empty_row_index, empty_row_index.size());
  }
};
} // end of namespace functor

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T, Tindex)			\
  template struct functor::ParallelSparseFillEmptyRows<GPUDevice, T, Tindex>;
#define EXPLICITLY_INSTANTIATE_FUNCTOR_TINDEX_INT64(T) EXPLICITLY_INSTANTIATE_FUNCTOR(T, int64)

TF_CALL_POD_TYPES(EXPLICITLY_INSTANTIATE_FUNCTOR_TINDEX_INT64)
#undef EXPLICITLY_INSTANTIATE_FUNCTOR_TINDEX_INT64
#undef EXPLICITLY_INSTANTIATE_FUNCTOR

} // end of namespace tensorflow

#endif // end of GOOGLE_CUDA

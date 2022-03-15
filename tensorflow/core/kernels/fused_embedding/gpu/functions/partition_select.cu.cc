#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/partition_select.cu.h"

#include <cub/cub.cuh>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace fused_embedding {

// A macro helper to declare SelectScanKernels, because they just have only
// a little bit differences. Need to define macros
#define DeclareSelectScanKernel                                            \
  template <typename T>                                                    \
  __global__ void SelectScanKernelName(                                    \
      const T* keys, SelectScanKernelArgs, const int64 num_partitions,     \
      const int64 length, const int64 predicates_length,                   \
      const int64 counters_length, unsigned int* predicates,               \
      unsigned int* counters) {                                            \
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;                     \
    int lnid = threadIdx.x % 32;                                           \
    int g_warp_id = g_tid >> 5;                                            \
    unsigned int mask;                                                     \
    unsigned int cnt;                                                      \
                                                                           \
    for (int j = 0; j < num_partitions; j++) {                             \
      SelectScanKernelLoadCodeBlock;                                       \
      _Pragma("unroll") for (int i = 0; i < 32; i++) {                     \
        int selected;                                                      \
        int load_offset = (g_warp_id << 10) + (i << 5) + lnid;             \
        if (load_offset < length) {                                        \
          T key = keys[load_offset];                                       \
          SelectScanKernelEvalCodeBlock;                                   \
        } else {                                                           \
          selected = 0;                                                    \
        }                                                                  \
                                                                           \
        mask = __ballot_sync(0xffffffff, selected);                        \
        if (lnid == 0)                                                     \
          predicates[j * predicates_length + (g_warp_id << 5) + i] = mask; \
        if (lnid == i) cnt = __popc(mask);                                 \
      }                                                                    \
      _Pragma("unroll") for (int offset = 16; offset > 0; offset >>= 1) {  \
        cnt += __shfl_down_sync(0xffffffff, cnt, offset);                  \
      }                                                                    \
                                                                           \
      if (lnid == 0) counters[j * counters_length + g_warp_id] = cnt;      \
    }                                                                      \
  }

#define DeclareSelectKernel                                                    \
  template <typename T, typename TIndex>                                       \
  __global__ void SelectKernelName(                                            \
      const T* keys, const TIndex* indices, SelectKernelArgs,                  \
      unsigned int* predicates, unsigned int* inclusive_sum_counters,          \
      const int64 num_partitions, const int64 length,                          \
      const int64 predicates_length, const int64 counters_length,              \
      void** output_ptrs) {                                                    \
    const int g_tid = blockIdx.x * blockDim.x + threadIdx.x;                   \
    const int lnid = threadIdx.x % 32;                                         \
    const int g_warp_id = g_tid >> 5;                                          \
    unsigned int predmask;                                                     \
    unsigned int cnt;                                                          \
    for (int j = 0; j < num_partitions; j++) {                                 \
      SelectKernelLoadCodeBlock;                                               \
      T* keys_output_ptr = (T*)(output_ptrs[2 * j]);                           \
      TIndex* indices_output_ptr = (TIndex*)(output_ptrs[2 * j + 1]);          \
      predmask = predicates[j * predicates_length + (g_warp_id << 5) + lnid];  \
      cnt = __popc(predmask);                                                  \
      _Pragma("unroll") for (int offset = 1; offset < 32; offset <<= 1) {      \
        int n = __shfl_up_sync(0xffffffff, cnt, offset);                       \
        if (lnid >= offset) cnt += n;                                          \
      }                                                                        \
      unsigned int global_index = 0;                                           \
      if (g_warp_id > 0) global_index = inclusive_sum_counters[g_warp_id - 1]; \
      _Pragma("unroll") for (int i = 0; i < 32; i++) {                         \
        unsigned int mask = __shfl_sync(0xffffffff, predmask, i);              \
        unsigned int sub_group_index = 0;                                      \
        if (i > 0) sub_group_index = __shfl_sync(0xffffffff, cnt, i - 1);      \
        if (mask & (1 << lnid)) {                                              \
          int load_offset = (g_warp_id << 10) + (i << 5) + lnid;               \
          T key =                                                              \
              keys[load_offset]; /* Will not cause out of boundry access,      \
                                   because mask will be 0 for this thread*/    \
          TIndex indice = indices[load_offset];                                \
          SelectKernelRecalcKeyCodeBlock;                                      \
          int output_offset = global_index + sub_group_index +                 \
                              __popc(mask & (((unsigned int)(1) << lnid) -     \
                                             (unsigned int)(1)));              \
          keys_output_ptr[output_offset] = new_key;                            \
          indices_output_ptr[output_offset] = indice;                          \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

// A macro helper to declare DefinePartitionSelect. Need to define following
// macros before using this: SelectScanKernelName,
// SelectArgs, SelectScanPassArgs
#define DeclareSelect                                                          \
  template <typename T, typename TIndex>                                       \
  void SelectName(OpKernelContext* ctx, const Tensor* keys,                    \
                  const Tensor& indices, SelectArgs,                           \
                  const int64 num_partitions, OpOutputList& selected_keys,     \
                  OpOutputList& selected_indices) {                            \
    OP_REQUIRES(ctx, keys->dims() == 1,                                        \
                errors::InvalidArgument("Tensor keys must ranks 1"));          \
    const GPUDevice& device = ctx->eigen_gpu_device();                         \
    const int64 length = keys->shape().dim_size(0);                            \
    cudaEvent_t memcpy_event;                                                  \
    cudaEventCreateWithFlags(&memcpy_event, cudaEventDisableTiming);           \
    Tensor predicates;                                                         \
    Tensor counters;                                                           \
    const int64 select_scan_warps =                                            \
        length % 1024 == 0 ? (length / 1024) : (length / 1024 + 1);            \
    const int64 counters_length = select_scan_warps;                           \
    const int64 predicates_length = select_scan_warps * 32;                    \
    OP_REQUIRES_OK(                                                            \
        ctx, ctx->allocate_temp(DT_UINT32,                                     \
                                TensorShape{counters_length * num_partitions}, \
                                &counters));                                   \
    OP_REQUIRES_OK(                                                            \
        ctx, ctx->allocate_temp(                                               \
                 DT_UINT32, TensorShape{predicates_length * num_partitions},   \
                 &predicates));                                                \
                                                                               \
    {                                                                          \
      const int64 threads = 32;                                                \
      const int64 blocks = select_scan_warps;                                  \
      OP_REQUIRES_OK(                                                          \
          ctx, GpuLaunchKernel(SelectScanKernelName<T>, blocks, threads, 0,    \
                               device.stream(), data_p_with_type<T>(keys),     \
                               SelectScanPassArgs, num_partitions, length,     \
                               predicates_length, counters_length,             \
                               data_p_with_type<unsigned int>(predicates),     \
                               data_p_with_type<unsigned int>(counters)));     \
    }                                                                          \
    Tensor inclusive_sum_counters;                                             \
    OP_REQUIRES_OK(                                                            \
        ctx, ctx->allocate_temp(DT_UINT32,                                     \
                                TensorShape{counters_length * num_partitions}, \
                                &inclusive_sum_counters));                     \
    size_t cub_tmp_storage_bytes;                                              \
    Tensor cub_tmp_storage;                                                    \
    cub::DeviceScan::InclusiveSum(                                             \
        nullptr, cub_tmp_storage_bytes,                                        \
        data_p_with_type<unsigned int>(counters),                              \
        data_p_with_type<unsigned int>(inclusive_sum_counters),                \
        counters_length, device.stream());                                     \
    OP_REQUIRES_OK(                                                            \
        ctx,                                                                   \
        ctx->allocate_temp(DT_INT8, TensorShape{int64(cub_tmp_storage_bytes)}, \
                           &cub_tmp_storage));                                 \
    for (int i = 0; i < num_partitions; i++) {                                 \
      cub::DeviceScan::InclusiveSum(                                           \
          data_p_with_type<void>(cub_tmp_storage), cub_tmp_storage_bytes,      \
          data_p_with_type<unsigned int>(counters) + i * counters_length,      \
          data_p_with_type<unsigned int>(inclusive_sum_counters) +             \
              i * counters_length,                                             \
          counters_length, device.stream());                                   \
    }                                                                          \
    std::vector<unsigned int> selected_nums_host;                              \
    selected_nums_host.resize(num_partitions);                                 \
    cudaMemcpy2DAsync(selected_nums_host.data(), 1 * sizeof(unsigned int),     \
                      data_p_with_type<unsigned int>(inclusive_sum_counters) + \
                          counters_length - 1,                                 \
                      counters_length * sizeof(unsigned int),                  \
                      1 * sizeof(unsigned int), num_partitions,                \
                      cudaMemcpyDeviceToHost, device.stream());                \
    cudaEventRecord(memcpy_event, device.stream());                            \
    cudaEventSynchronize(memcpy_event);                                        \
                                                                               \
    std::vector<void*> output_ptrs_host;                                       \
    output_ptrs_host.resize(2 * num_partitions);                               \
    for (int i = 0; i < num_partitions; i++) {                                 \
      Tensor* tmp_out;                                                         \
      OP_REQUIRES_OK(                                                          \
          ctx, selected_keys.allocate(                                         \
                   i, TensorShape({int64(selected_nums_host[i])}), &tmp_out)); \
      output_ptrs_host[2 * i] = data_p_with_type<void>(tmp_out);               \
      OP_REQUIRES_OK(                                                          \
          ctx, selected_indices.allocate(                                      \
                   i, TensorShape({int64(selected_nums_host[i])}), &tmp_out)); \
      output_ptrs_host[2 * i + 1] = data_p_with_type<void>(tmp_out);           \
    }                                                                          \
    Tensor output_ptrs;                                                        \
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(                                    \
                            DT_UINT64, TensorShape{int64(2 * num_partitions)}, \
                            &output_ptrs));                                    \
    cudaMemcpyAsync(data_p_with_type<void>(output_ptrs),                       \
                    output_ptrs_host.data(),                                   \
                    2 * num_partitions * sizeof(size_t),                       \
                    cudaMemcpyHostToDevice, device.stream());                  \
    {                                                                          \
      const int64 threads = 32;                                                \
      const int64 blocks = select_scan_warps;                                  \
      OP_REQUIRES_OK(                                                          \
          ctx, GpuLaunchKernel(                                                \
                   SelectKernelName<T, TIndex>, blocks, threads, 0,            \
                   device.stream(), data_p_with_type<T>(keys),                 \
                   data_p_with_type<TIndex>(indices), SelectPassArgs,          \
                   data_p_with_type<unsigned int>(predicates),                 \
                   data_p_with_type<unsigned int>(inclusive_sum_counters),     \
                   num_partitions, length, predicates_length, counters_length, \
                   data_p_with_type<void*>(output_ptrs)));                     \
    }                                                                          \
  }

#define SelectName PartitionSelectDiv
#define SelectArgs const Tensor& accu_div
#define SelectScanPassArgs data_p_with_type<int64>(accu_div)
#define SelectPassArgs SelectScanPassArgs

#define SelectScanKernelName SelectScanDivKernel
#define SelectScanKernelArgs const int64* accu_div
#define SelectScanKernelLoadCodeBlock              \
  int64 lower_bound = j > 0 ? accu_div[j - 1] : 0; \
  int64 upper_bound = accu_div[j];
#define SelectScanKernelEvalCodeBlock \
  selected = int(key >= lower_bound && key < upper_bound);

#define SelectKernelName SelectDivKernel
#define SelectKernelArgs SelectScanKernelArgs

#define SelectKernelLoadCodeBlock \
  int64 lower_bound = j > 0 ? accu_div[j - 1] : 0;
#define SelectKernelRecalcKeyCodeBlock T new_key = key - lower_bound;

DeclareSelectScanKernel;
DeclareSelectKernel;
DeclareSelect;
template void PartitionSelectDiv<int64, int64>(
    OpKernelContext* ctx, const Tensor* keys, const Tensor& indices,
    const Tensor& accu_div, const int64 num_partitions,
    OpOutputList& selected_keys, OpOutputList& selected_indices);

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
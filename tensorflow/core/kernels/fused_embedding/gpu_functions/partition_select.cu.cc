#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/fused_embedding/gpu_functions/partition_select.cu.h"

#include <cub/cub.cuh>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/kernels/fused_embedding/fused_embedding.cu.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace fused_embedding {

// A macro helper to declare SelectScanKernels, because they just have only
// a little bit differences. Need to define following macros before using this:
// KernelName, SelectScanKernelArgs, PartitionConditionLoadCodeBlock,
// PartitionSelectCodeBlock
#define DeclareSelectScanKernel                                           \
  template <typename T>                                                   \
  __global__ void KernelName(                                             \
      const T* keys, SelectScanKernelArgs, const int64 num_partitions,    \
      const int64 length, const int64 predicates_length,                  \
      const int64 counters_length, unsigned int* predicates,              \
      unsigned int* counters) {                                           \
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;                    \
    int lnid = threadIdx.x % 32;                                          \
    if (g_tid >= (length >> 5)) return;                                   \
    int g_warp_id = g_tid >> 5;                                           \
    unsigned int mask;                                                    \
    unsigned int cnt;                                                     \
                                                                          \
    for (int j = 0; j < num_partitions; j++) {                            \
      PartitionConditionLoadCodeBlock;                                    \
      _Pragma("unroll") for (int i = 0; i < 32; i++) {                    \
        int selected;                                                     \
        int load_offset = (g_warp_id << 10) + (i << 5) + lnid;            \
        if (load_offset < length) {                                       \
          T key = keys[load_offset];                                      \
          PartitionSelectCodeBlock;                                       \
        } else {                                                          \
          selected = 0;                                                   \
        }                                                                 \
                                                                          \
        mask = __ballot(selected);                                        \
        if (lnid == 0) predicates[(g_warp_id << 5)] = mask;               \
        if (lnid == i) cnt = __popc(mask);                                \
      }                                                                   \
      _Pragma("unroll") for (int offset = 16; offset > 0; offset >>= 1) { \
        cnt += __shfl_down(cnt, offset);                                  \
      }                                                                   \
                                                                          \
      if (lnid == 0) counters[g_warp_id] = cnt;                           \
    }                                                                     \
  }

// A macro helper to declare DefinePartitionSelect. Need to define following
// macros before using this: Name, KernelName, PartitionSelectArgs,
// PartitionSelectPassArgs
#define DeclarePartitionSelect                                                 \
  template <typename T, typename TIndex>                                       \
  void Name(OpKernelContext* ctx, const Tensor& keys, PartitionSelectArgs,     \
            const int64 num_partitions) {                                      \
    OP_REQUIRES(ctx, keys.dims() == 1,                                         \
                errors::InvalidArgument("Tensor keys must ranks 1"));          \
    const GPUDevice& device = ctx->eigen_gpu_device();                         \
    const int64 length = keys.shape().dim_size(0);                             \
    cudaEvent_t memcpy_event;                                                  \
    cudaEventCreateWithFlags(&memcpy_event, cudaEventDisableTiming);           \
    Tensor predicates;                                                         \
    Tensor counters;                                                           \
    const int64 select_scan_warps =                                            \
        length % 1024 == 0 ? (length / 1024) : (length / 1024 + 1);            \
    const int64 counters_length = select_scan_warps;                           \
    const int64 predicates_length = select_scan_warps * 32;                    \
    {                                                                          \
      const int64 threads = 32;                                                \
      const int64 blocks = select_scan_warps;                                  \
      OP_REQUIRES_OK(                                                          \
          ctx, ctx->allocate_temp(                                             \
                   DT_UINT32, TensorShape{predicates_length * num_partitions}, \
                   &predicates));                                              \
                                                                               \
      OP_REQUIRES_OK(                                                          \
          ctx, ctx->allocate_temp(                                             \
                   DT_UINT32, TensorShape{counters_length * num_partitions},   \
                   &counters));                                                \
      OP_REQUIRES_OK(                                                          \
          ctx, GpuLaunchKernel(KernelName<T>, blocks, threads, 0,              \
                               device.stream(), data_p_with_type<T*>(keys),    \
                               PartitionSelectPassArgs, num_partitions,        \
                               length, predicates_length, counters_length,     \
                               data_p_with_type<unsigned int*>(predicates),    \
                               data_p_with_type<unsigned int*>(counters)));    \
    }                                                                          \
    Tensor selected_sums;                                                      \
    OP_REQUIRES_OK(ctx,                                                        \
                   ctx->allocate_temp(DT_UINT32, TensorShape{num_partitions},  \
                                      &selected_sums));                        \
    size_t cub_tmp_storage_bytes;                                              \
    Tensor cub_tmp_storage;                                                    \
    cub::DeviceReduce::Sum(nullptr, cub_tmp_storage_bytes,                     \
                           data_p_with_type<unsigned int*>(counters),          \
                           data_p_with_type<unsigned int*>(selected_sums),     \
                           counters_length, device.stream());                  \
    OP_REQUIRES_OK(                                                            \
        ctx, ctx->allocate_temp(DT_INT8, TensorShape{cub_tmp_storage_bytes},   \
                                &cub_tmp_storage));                            \
    cub::DeviceReduce::Sum(data_p_with_type<void*>(cub_tmp_storage),           \
                           cub_tmp_storage_bytes,                              \
                           data_p_with_type<unsigned int*>(counters),          \
                           data_p_with_type<unsigned int*>(selected_sums),     \
                           counters_length, device.stream());                  \
    for (int i = 0; i < num_partitions; i++) {                                 \
      cub::DeviceReduce::Sum(                                                  \
          data_p_with_type<void*>(cub_tmp_storage), cub_tmp_storage_bytes,     \
          data_p_with_type<unsigned int*>(counters) + i * counters_length,     \
          data_p_with_type<unsigned int*>(selected_sums) + i, counters_length, \
          device.stream());                                                    \
    }                                                                          \
    std::vector<unsigned int> selected_sums_host;                              \
    selected_sums_host.resize(num_partitions);                                 \
    cudaMemcpyAsync(selected_sums_host.data(),                                 \
                    data_p_with_type<unsigned int*>(selected_sums),            \
                    sizeof(unsigned int) * num_partitions,                     \
                    cudaMemcpyDeviceToHost, device.stream());                  \
    cudaEventRecord(memcpy_event, device.stream());                            \
    cudaEventSynchronize(memcpy_event);                                        \
  }

#define Name PartitionSelectDiv
#define PartitionSelectArgs const Tensor& accu_div
#define PartitionSelectPassArgs data_p_with_type<int64_t*>(accu_div)
#define KernelName SelectScanDivKernel
#define SelectScanKernelArgs const int64_t* accu_div
#define PartitionConditionLoadCodeBlock              \
  int64_t lower_bound = j > 0 ? accu_div[j - 1] : 0; \
  int64_t upper_bound = accu_div[j];
#define PartitionSelectCodeBlock \
  selected = int(key >= lower_bound && key < upper_bound);
DeclareSelectScanKernel;
DeclarePartitionSelect;

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
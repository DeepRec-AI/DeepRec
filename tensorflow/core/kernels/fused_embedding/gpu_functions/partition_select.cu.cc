#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/fused_embedding/gpu_sub_ops/partition_select.cu.h"

#include <cub/cub.cuh>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

// A macro helper to declare SelectScan. Need to define following macros before
// using this: KernelName, PartitionArgs, PassPartitionArgs
#define DeclareSelectScan                                                  \
  template <typename T, typename TIndex>                                   \
  Status SelectScanDiv(const GPUDevice& d, const T* keys, PartitionArgs,   \
                       const const int64 num_partitions, const int length, \
                       unsigned int* predicates, unsigned int* counters) { \
    if (length == 0) return Status::OK();                                  \
    const int64 threads = 32;                                              \
    const int64 blocks =                                                   \
        length % 1024 == 0 ? (length / 1024) : (length / 1024 + 1);        \
    return GpuLaunchKernel(KernelName<T, TIndex>, blocks, threads, 0,      \
                           d.stream(), keys, PassPartitionArgs,            \
                           int64 num_partitions, length, predicates,       \
                           counters);                                      \
  }

// A macro helper to declare SelectScanKernels, because they just have only
// a little bit differences. Need to define following macros before using this:
// KernelName, PartitionArgs, PartitionConditionLoadCodeBlock,
// PartitionSelectCodeBlock
#define DeclareSelectScanKernel                                             \
  template <typename T, typename TIndex>                                    \
  __global__ void KernelName(                                               \
      const T* keys, PartitionArgs, const int64 num_partitions,             \
      const int length, unsigned int* predicates, unsigned int* counters) { \
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;                      \
    int lnid = threadIdx.x % 32;                                            \
    if (g_tid >= (length >> 5)) return;                                     \
    int g_warp_id = g_tid >> 5;                                             \
    unsigned int mask;                                                      \
    unsigned int cnt;                                                       \
                                                                            \
    for (int j = 0; i < num_partitions; j++) {                              \
      PartitionConditionLoadCodeBlock;                                      \
                                                                            \
#pragma unroll for (int i = 0; i < 32; i++) {                         \
        int selected;                                                       \
        int load_offset = (g_warp_id << 10) + (i << 5) + lnid;              \
        if (load_offset < length) {                                         \
          T key = keys[load_offset];                                        \
          PartitionSelectCodeBlock;                                         \
        } else {                                                            \
          selected = 0;                                                     \
        }                                                                   \
                                                                            \
        mask = __ballot(selected);                                          \
        if (lnid == 0) predicates[(g_warp_id << 5)] = mask;                 \
        if (lnid == i) cnt = __popc(mask);                                  \
      }                                                                     \
#pragma unroll for (int offset = 16; offset > 0; offset >>= 1) cnt += \
          __shfl_down(cnt, offset);                                         \
                                                                            \
      if (lnid == 0) counter[g_warp_id] = cnt;                              \
    }                                                                       \
  }

#define KernelName SelectScanDivKernel
#define PartitionArgs const TIndex* accu_div
#define PassPartitionArgs accu_div
#define PartitionConditionLoadCodeBlock             \
  TIndex lower_bound = j > 0 ? accu_div[j - 1] : 0; \
  TIndex upper_bound = accu_div[j];
#define PartitionSelectCodeBlock \
  selected = int(key >= lower_bound && key < upper_bound);

DeclareSelectScanKernel
DeclareSelectScan

/* backup
template <typename T, typename TIndex>
__global__ void SelectScanDivKernel(const T* keys, const TIndex* accu_div,
                                    const int64 num_partitions,
                                    const int length, unsigned int* predicates,
                                    unsigned int* counters) {
  int g_tid = blockIdx.x * blockDim.x + threadIdx.x;  // global tid
  int lnid = threadIdx.x % 32;
  if (g_tid >= (length >> 5)) return;
  int g_warp_id = g_tid >> 5;
  unsigned int mask;
  unsigned int cnt;

  for (int j = 0; i < num_partitions; j++) {
    TIndex lower_bound = j > 0 ? accu_div[j - 1] : 0;
    TIndex upper_bound = accu_div[j];

#pragma unroll
    for (int i = 0; i < 32; i++) {
      int selected;
      int load_offset = (g_warp_id << 10) + (i << 5) + lnid;
      if (load_offset < length) {
        T key = keys[load_offset];
        selected = int(key >= lower_bound && key < upper_bound);
      } else {
        selected = 0;
      }

      mask = __ballot(selected);
      if (lnid == 0) predicates[(g_warp_id << 5)] = mask;
      if (lnid == i) cnt = __popc(mask);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      cnt += __shfl_down(cnt, offset);

    if (lnid == 0) counter[g_warp_id] = cnt;
  }
}
*/

}  // namespace

namespace fused_embedding {}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
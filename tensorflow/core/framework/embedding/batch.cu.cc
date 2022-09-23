#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV

#include "tensorflow/core/framework/embedding/batch.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
template<class V>
__global__ void BatchCopy(V** batch, V* val_base, int value_len,
    int limit, V** default_value, bool* init_flags) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (i < limit * value_len) {
    val_base[i] = *(batch[item_id] + item_pos);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T) \
   template __global__ void BatchCopy<T>(T**, T*, int, int, T**, bool*);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void BatchUnpack(V** dev_value_address,
    V* memcpy_buffer_gpu, int value_len, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (i < limit * value_len) {
    *(dev_value_address[item_id] + item_pos) = memcpy_buffer_gpu[i];
  }
}

template __global__ void BatchUnpack<int>(
    int**, int*, int, int);
template __global__ void BatchUnpack<float>(
    float**, float*, int, int);
template __global__ void BatchUnpack<double>(
    double**, double*, int, int);
template __global__ void BatchUnpack<long long>(
    long long**, long long*, int, int);

template<class V>
__global__ void SparseApplyAdagradGPU(V** a, V** v, V* g, float lr,
    int embedding_dim, long long int limit, bool* init_flags,
    V* default_value) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    *(a[item_id] + item_pos) += g[i] * g[i];
    *(v[item_id] + item_pos) -=
        lr * g[i] * rsqrt(*(a[item_id] + item_pos));
  }
}

template __global__ void SparseApplyAdagradGPU<float>(
    float**, float**, float*, float, int, long long int, bool*, float*);
template __global__ void SparseApplyAdagradGPU<double>(
    double**, double**, double*, float, int, long long int, bool*, double*);

template<class V>
__global__ void CopyEmbedding(V** batch, V** batch_data_space,
    int total_dims, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / total_dims;
  int item_pos = i % total_dims;

  if (i < limit  * total_dims) {
    *(batch_data_space[item_id] + item_pos) = *(batch[item_id] + item_pos);
  }
}

template __global__ void CopyEmbedding<int>(
    int**, int**, int, int);
template __global__ void CopyEmbedding<float>(
    float**, float**, int, int);
template __global__ void CopyEmbedding<double>(
    double**, double**, int, int);
template __global__ void CopyEmbedding<long long>(
    long long**, long long**, int, int);

}  // namespace tensorflow
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

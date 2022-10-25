#include <algorithm>
#include <numeric>
#include <vector>
#include <bitset>
#include <time.h>
#include <sys/time.h>
#if defined (__AVX2__)
#include <immintrin.h>
#endif
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "absl/strings/str_join.h"

namespace tensorflow {

using namespace std;

template <typename T>
struct LogicalSumFunctor {
  static EIGEN_ALWAYS_INLINE Status
  Compute(OpKernelContext* context, const Tensor& q_in, const Tensor& k_in,
                        Tensor* out) {
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    auto s = q_in.dim_size(q_in.dims() - 1);
    auto task_num     = std::max(q_in.NumElements(),k_in.NumElements()) / s;

    const auto bcast_size   = (q_in.dim_size(1) == k_in.dim_size(1)) ? 
                          1 : std::max(q_in.dim_size(1),k_in.dim_size(1));
    const auto q_in_ptr = q_in.flat<float>().data();
    const auto k_in_ptr = k_in.flat<float>().data();
    auto out_ptr = out->flat<T>().data();

    auto DoCompute = [&](int start, int limit) {
      int32_t j = start/bcast_size;
      if (q_in.dim_size(1) >= k_in.dim_size(1)){
        for (int32_t i = start;  i < limit; i++) {
          j = i / bcast_size;
          ComputeUnit(context, q_in_ptr + i * s, k_in_ptr + j * s, s, out_ptr + i);
        }
      } else {
        for (int32_t i = start;  i < limit; i++) {
          j = i / bcast_size;
          ComputeUnit(context, q_in_ptr + j * s, k_in_ptr + i * s, s, out_ptr + i);
        }
      }
    };
    const int64_t unit_cost = 6 * s; //evalation cost
    Shard(worker_threads.num_threads, worker_threads.workers, task_num,
          unit_cost, DoCompute);
    return Status::OK();
    
  }

  static EIGEN_ALWAYS_INLINE Status
  ComputeUnitNormal(OpKernelContext* context, const float* q_in,
              const float* k_in,
              int32_t unit_size, T* out) {
    int16_t sum = 0;
#if defined (__AVX2__)
    int32 unit_size_batch = unit_size - unit_size % 32;
    __m256i q_256_0, q_256_1, q_256_2, q_256_3, k_256_0, k_256_1, k_256_2, k_256_3;
    __m256 xor_result_0, xor_result_1, xor_result_2, xor_result_3;
    int sign_bits_0, sign_bits_1, sign_bits_2, sign_bits_3;
    for (int64 i = 0; i < unit_size_batch; i += 32) {
      // load 8 float
      q_256_0 = _mm256_castps_si256(_mm256_loadu_ps(q_in + i));
      q_256_1 = _mm256_castps_si256(_mm256_loadu_ps(q_in + i + 8));
      q_256_2 = _mm256_castps_si256(_mm256_loadu_ps(q_in + i + 16));
      q_256_3 = _mm256_castps_si256(_mm256_loadu_ps(q_in + i + 24));
      k_256_0 = _mm256_castps_si256(_mm256_loadu_ps(k_in + i));
      k_256_1 = _mm256_castps_si256(_mm256_loadu_ps(k_in + i + 8));
      k_256_2 = _mm256_castps_si256(_mm256_loadu_ps(k_in + i + 16));
      k_256_3 = _mm256_castps_si256(_mm256_loadu_ps(k_in + i + 24));

      // do xor for 8 float
      xor_result_0 = _mm256_castsi256_ps(_mm256_xor_si256(q_256_0, k_256_0));
      xor_result_1 = _mm256_castsi256_ps(_mm256_xor_si256(q_256_1, k_256_1));
      xor_result_2 = _mm256_castsi256_ps(_mm256_xor_si256(q_256_2, k_256_2));
      xor_result_3 = _mm256_castsi256_ps(_mm256_xor_si256(q_256_3, k_256_3));

      // get sign bits of xor_result
      sign_bits_0 = _mm256_movemask_ps(xor_result_0);
      sign_bits_1 = _mm256_movemask_ps(xor_result_1);
      sign_bits_2 = _mm256_movemask_ps(xor_result_2);
      sign_bits_3 = _mm256_movemask_ps(xor_result_3);

      // add result to sum
      sum += std::bitset<10>(sign_bits_0).count();
      sum += std::bitset<10>(sign_bits_1).count();
      sum += std::bitset<10>(sign_bits_2).count();
      sum += std::bitset<10>(sign_bits_3).count();
    }

    for (int64 i = unit_size_batch; i < unit_size; ++i) {
      auto *q_in_c = reinterpret_cast<const unsigned char *>(&q_in[i]);
      auto *k_in_c = reinterpret_cast<const unsigned char *>(&k_in[i]);
      auto q_sign = q_in_c[sizeof(float)-1] >> 7;
      auto k_sign = k_in_c[sizeof(float)-1] >> 7;
      sum += q_sign ^ k_sign;
    }
#else
    for (int32_t i = 0; i < unit_size; i++) {
      auto *q_in_c = reinterpret_cast<const unsigned char *>(&q_in[i]);
      auto *k_in_c = reinterpret_cast<const unsigned char *>(&k_in[i]);
      auto q_sign = q_in_c[sizeof(float)-1] >> 7;
      auto k_sign = k_in_c[sizeof(float)-1] >> 7;
      sum += q_sign ^ k_sign;
    }
#endif
    *out = static_cast<T>(-sum);
    return Status::OK();
  }

  static EIGEN_ALWAYS_INLINE Status
  ComputeUnit(OpKernelContext* context, const float* q_in,
              const float* k_in,
              int32_t unit_size, T* out) {

    return ComputeUnitNormal(context, q_in, k_in, unit_size, out); 

  }
};

template <typename T>
class LogicalSum : public OpKernel {
 public:
  explicit LogicalSum(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, num_inputs() == 2,
                errors::InvalidArgument("input number == 2, got ",
                                        num_inputs()));
    const auto& q_in = context->input(0);
    OP_REQUIRES(context, q_in.dims() == 3,
                  errors::InvalidArgument("q_in must be == 3-D, got shape ",
                                        q_in.shape().DebugString()));
    const auto& k_in = context->input(1);
    OP_REQUIRES(context, k_in.dims() == 3,
                  errors::InvalidArgument("k_in must be >= 3-D, got shape ",
                                        k_in.shape().DebugString()));

    OP_REQUIRES(context, q_in.dim_size(0) == k_in.dim_size(0),
                  errors::InvalidArgument("q_in dim 0 size must be equal to k_in, got q shape ",
                                        q_in.shape().DebugString(), ", got k shape ", k_in.shape().DebugString()));

    OP_REQUIRES(context, q_in.dim_size(2) == k_in.dim_size(2),
                  errors::InvalidArgument("q_in dim 2 size must be equal to k_in, got q shape ",
                                        q_in.shape().DebugString(), ", got k shape ", k_in.shape().DebugString()));

    OP_REQUIRES(context, (q_in.shape() == k_in.shape() || q_in.dim_size(1) == 1 || k_in.dim_size(1)==1 ),
                  errors::InvalidArgument("q_in shape  must be equal to k_in, or the 2-dim can broadcast, got q shape ",
                                        q_in.shape().DebugString(), ", got k shape ", k_in.shape().DebugString()));


    Tensor* output = nullptr;
    TensorShape out_shape;
    out_shape.AddDim(std::max(q_in.dim_size(0),k_in.dim_size(0)));
    out_shape.AddDim(std::max(q_in.dim_size(1),k_in.dim_size(1)));

    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    Status s = LogicalSumFunctor<T>::Compute(
        context, q_in, k_in, output);

    OP_REQUIRES_OK(context, s);
  }
};

REGISTER_KERNEL_BUILDER(Name("LogicalSum")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    LogicalSum<float>);

}  // namespace tensorflow

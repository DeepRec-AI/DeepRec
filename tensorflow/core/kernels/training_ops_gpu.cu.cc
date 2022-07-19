/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/training_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

#if TENSORFLOW_USE_ROCM

#include "rocm/include/hip/hip_complex.h"

#endif  // TENSORFLOW_USE_ROCM

// if any kernels involving complex sqrt/rsqrt are compiled with ROCm, build
// process completes without errors,but the resulting executable ends up
// unusable (throwing errors "no device code available for function" for
/// completely unrelated kernels.)
// We also can't cast to hipFloatComplex etc. because (as of 2020-01) HIP does
// not provide sqrt for complex.
// We have no choice but to implement sqrt and rsqrt by hand
template <typename T>
__device__ T impl_sqrt(T x) {
  return sqrt(x);
}
template <typename T>
__device__ T impl_rsqrt(T x) {
  return rsqrt(x);
}
template <>
__device__ Eigen::half impl_sqrt(Eigen::half x) {
  return __float2half(sqrt(__half2float(x)));
}
template <>
__device__ Eigen::half impl_rsqrt(Eigen::half x) {
  return __float2half(rsqrt(__half2float(x)));
}

template <class T>
__device__ std::complex<T> impl_sqrt(std::complex<T> x) {
  T re = x.real(), im = x.imag();
  T mod_x = sqrt(re * re + im * im);
  const T root2 = 0.7071067811865475;
  // We pick the root with the same sign of the imaginary component as
  // the input.
  T root[2] = {T(sqrt(mod_x + re) * root2),
               T(sqrt(mod_x - re) * root2 * (im >= 0 ? 1. : -1.))};
  // hcc/clang is really weird with its support of complex in device code;
  // for some reason it does not permit a 2-argument constructor
  return *(reinterpret_cast<std::complex<T>*>(&root));
}

template <class T>
__device__ T rsqrt_helper(T x) {
  return 0.5 * x + 0.125 * x * x + 0.0625 * x * x * x;
}

template <class T>
__device__ std::complex<T> impl_rsqrt(std::complex<T> x) {
  T re = x.real(), im = x.imag();
  T r = rsqrt(re * re + im * im);
  T ar2 = re * r * r;
  const T root2 = 0.7071067811865475;
  T root[2];
  // With float, calculating 1+re*r and 1-re*r may result in excessive errors
  // due to subtraction of two close values. We have to get fancy
  root[0] = sqrt(r * ((std::is_same<T, float>::value && re * r < -0.98)
                          ? rsqrt_helper(im * im * r * r)
                          : max(T(0.0), 1 + re * r))) *
            root2;
  root[1] = sqrt(r * ((std::is_same<T, float>::value && re * r > 0.98)
                          ? rsqrt_helper(im * im * r * r)
                          : max(T(0.0), 1 - re * r))) *
            root2 * (im >= 0 ? -1. : 1.);
  return *(reinterpret_cast<std::complex<T>*>(&root));
}

template <typename T>
__device__ T impl_fabs(T x) {
  return fabs(x);
}
template <>
__device__ Eigen::half impl_fabs(Eigen::half x) {
  return __float2half(fabs(__half2float(x)));
}

template <typename T>
__device__ T impl_sign(T x) {
  return x == T(0) ? T(0) : x < T(0) ? T(-1) : T(1);
}

template <typename T, typename Tindex, bool has_epsilon>
__global__ __launch_bounds__(1024) void SparseApplyAdagradKernel(
    T* var, T* accum, const T* lr, const T* epsilon, const T* grad,
    const Tindex* indices, Tindex param_rows, Tindex updates_size,
    Tindex indices_size, bool update_slots) {
  Tindex col_size = updates_size / indices_size;
  GPU_1D_KERNEL_LOOP(grad_index, updates_size) {
    Tindex indices_row = grad_index / col_size;
    Tindex param_row = indices[indices_row];
    if (param_row < 0 || param_row >= param_rows) {
      // Ignore indices that are out of range.
      continue;
    }

    // Compute the index of var and accum.
    Tindex param_index = param_row * col_size + (grad_index % col_size);

    // Read variables.
    T var_i = var[param_index];
    T accum_i = accum[param_index];
    T grad_i = grad[grad_index];
    const T lr_t = *lr;
    const T epsilon_t = *epsilon;

    if (update_slots) {
      accum_i += grad_i * grad_i;
    }
    if (has_epsilon) {
      var_i -= lr_t * grad_i / (sqrt(accum_i) + epsilon_t);
    } else {
      var_i -= lr_t * grad_i * impl_rsqrt(accum_i);
    }

    // Write update back to variables.
    var[param_index] = var_i;
    accum[param_index] = accum_i;
  }
}

template <typename T, typename Tindex>
__global__ __launch_bounds__(1024) void SparseApplyProximalAdagradKernel(
    T* var, T* accum, const T* lr, const T* l1, const T* l2, const T* grad,
    const Tindex* indices, Tindex param_rows, Tindex updates_size,
    Tindex indices_size) {
  Tindex col_size = updates_size / indices_size;
  GPU_1D_KERNEL_LOOP(grad_index, updates_size) {
    Tindex indices_row = grad_index / col_size;
    Tindex param_row = indices[indices_row];
    if (param_row < 0 || param_row >= param_rows) {
      // Ignore indices that are out of range.
      continue;
    }

    // Compute the index of var and accum.
    Tindex param_index = param_row * col_size + (grad_index % col_size);

    // Read variables.
    T var_i = var[param_index];
    T accum_i = accum[param_index];
    T grad_i = grad[grad_index];
    const T lr_t = *lr;
    const T l1_t = *l1;
    const T l2_t = *l2;

    accum_i += grad_i * grad_i;
    T learning_rate = lr_t * impl_rsqrt(accum_i);
    // compute v = w - lr * grad.
    T prox_var_i = var_i - grad_i * learning_rate;
    // compute sign(v) * max(|v| - lr * max(l1, 0), 0)
    var_i = (prox_var_i >= T(0) ? T(1.) : T(-1.)) *
            max(abs(prox_var_i) - learning_rate * max(l1_t, T(0)), T(0)) /
            (T(1.) + l2_t * learning_rate);

    // Write update back to variables.
    var[param_index] = var_i;
    accum[param_index] = accum_i;
  }
}

template <typename T, typename Tindex, bool has_l2_shrinkage>
__global__ void SparseApplyFtrlKernel(T* var, T* accum, T* linear, const T* lr,
                                      const T* l1, const T* l2,
                                      const T* l2_shrinkage, const T* lr_power,
                                      const T* grad, const Tindex* indices,
                                      Tindex param_rows, Tindex updates_size,
                                      Tindex indices_size,
                                      bool multiply_linear_by_lr) {
  const Tindex col_size = updates_size / indices_size;
  GPU_1D_KERNEL_LOOP(grad_index, updates_size) {
    const Tindex indices_row = grad_index / col_size;
    const Tindex param_row = indices[indices_row];
    if (param_row < 0 || param_row >= param_rows) {
      // Ignore indices that are out of range.
      continue;
    }

    // Compute the index of var and accum.
    const Tindex param_index = param_row * col_size + (grad_index % col_size);

    // Read variables.
    T var_i = var[param_index];
    T accum_i = accum[param_index];
    T linear_i = linear[param_index];
    const T grad_i = grad[grad_index];
    const T lr_t = *lr;
    const T l1_t = *l1;
    const T l2_t = *l2;
    const T lr_power_t = *lr_power;

    const T grad_shr_i =
        has_l2_shrinkage ? grad_i + static_cast<T>(2) * (*l2_shrinkage) * var_i
                         : grad_i;
    const T new_accum_i = accum_i + grad_i * grad_i;
    const bool lr_power_is_neg_half = lr_power_t == static_cast<T>(-0.5);
    const T pow_new_accum = lr_power_is_neg_half
                                ? sqrt(new_accum_i)
                                : pow(new_accum_i, -lr_power_t);
    const T pow_accum =
        lr_power_is_neg_half ? sqrt(accum_i) : pow(accum_i, -lr_power_t);
    T linear_change = grad_shr_i * lr_t - (pow_new_accum - pow_accum) * var_i;
    if (!multiply_linear_by_lr) {
      linear_change /= lr_t;
    }
    linear_i += linear_change;

    T l1_mult = l1_t;
    if (multiply_linear_by_lr) {
      l1_mult *= lr_t;
    }
    const T l1_reg_adjust = max(min(linear_i, l1_mult), -l1_mult);
    const T x = l1_reg_adjust - linear_i;
    T y = pow_new_accum + static_cast<T>(2) * l2_t * lr_t;
    if (!multiply_linear_by_lr) {
      y /= lr_t;
    }
    var_i = x / y;
    accum_i = new_accum_i;

    // Write update back to variables.
    var[param_index] = var_i;
    accum[param_index] = accum_i;
    linear[param_index] = linear_i;
  }
}

template <typename T>
struct ApplyGradientDescent<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    var.device(d) -= lr.reshape(single).broadcast(bcast) * grad;
  }
};

template <typename T>
struct ApplyAdagrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad, bool update_slots) {
    if (update_slots) {
      accum.device(d) += grad.square();
    }
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    var.device(d) -= lr.reshape(single).broadcast(bcast) * grad * accum.rsqrt();
  }
};

template <typename T>
struct ApplyAdagradV2<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool update_slots) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    if (update_slots) {
      accum.device(d) += grad.square();
    }
    const auto update =
        grad / (accum.sqrt() + epsilon.reshape(single).broadcast(bcast));
    var.device(d) -= lr.reshape(single).broadcast(bcast) * update;
  }
};

template <typename T, typename Tindex, bool has_epsilon>
struct SparseApplyAdagrad<GPUDevice, T, Tindex, has_epsilon> {
  Status operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar epsilon,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,
                    bool update_slots) {
    const Tindex N = static_cast<Tindex>(indices.dimension(0));
    if (N == 0) return Status::OK();

    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();
    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    return GpuLaunchKernel(
        SparseApplyAdagradKernel<T, Tindex, has_epsilon>, config.block_count,
        config.thread_per_block, 0, d.stream(), var.data(), accum.data(),
        lr.data(), epsilon.data(), grad.data(), indices.data(), first_dim_size,
        grad_size, indices_size, update_slots);
  }
};

template <typename T>
struct ApplyProximalAdagrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    // Fobos update per paper with Adagrad learning rate.
    accum.device(d) += grad.square();
    // Adagrad learning rate.
    // The following is the GPU equivalent of the CPU version:
    // auto learning_rate = accum.constant(lr()) * accum.rsqrt();
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto l1_bcast = l1.reshape(single).broadcast(bcast);
    auto l2_bcast = l2.reshape(single).broadcast(bcast);
    auto learning_rate = lr_bcast * accum.rsqrt();
    auto prox_var = var;
    // compute v = w - lr * grad.
    prox_var.device(d) -= grad * learning_rate;
    // compute sign(v) * max(|v| - lr * max(l1, 0), 0)
    var.device(d) = prox_var.sign() *
                    (prox_var.abs() - learning_rate * l1_bcast.cwiseMax(T(0.f)))
                        .cwiseMax(T(0.f)) /
                    (var.constant(T(1.f)) + l2_bcast * learning_rate);
  }
};

template <typename T, typename Tindex>
struct SparseApplyProximalAdagrad<GPUDevice, T, Tindex> {
  Status operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar l1,
                    typename TTypes<T>::ConstScalar l2,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstVec indices,
                    int64 inner_dim) {
    const Tindex N = static_cast<Tindex>(indices.dimension(0));
    if (N == 0) return Status::OK();

    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();
    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    return GpuLaunchKernel(SparseApplyProximalAdagradKernel<T, Tindex>,
                           config.block_count, config.thread_per_block, 0,
                           d.stream(), var.data(), accum.data(), lr.data(),
                           l1.data(), l2.data(), grad.data(), indices.data(),
                           first_dim_size, grad_size, indices_size);
  }
};

template <typename T>
struct ApplyAdadelta<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat accum_update,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    accum.device(d) = accum * rho.reshape(single).broadcast(bcast) +
                      grad.square() * (grad.constant(T(1)) -
                                       rho.reshape(single).broadcast(bcast));
    const auto update =
        (accum_update + epsilon.reshape(single).broadcast(bcast)).sqrt() *
        (accum + epsilon.reshape(single).broadcast(bcast)).rsqrt() * grad;
    var.device(d) -= update * lr.reshape(single).broadcast(bcast);
    accum_update.device(d) =
        accum_update * rho.reshape(single).broadcast(bcast) +
        update.square() *
            (grad.constant(T(1)) - rho.reshape(single).broadcast(bcast));
  }
};

template <typename T, typename Tindex, bool has_l2_shrinkage>
struct SparseApplyFtrl<GPUDevice, T, Tindex, has_l2_shrinkage> {
  Status operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::Matrix linear,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar l1,
                    typename TTypes<T>::ConstScalar l2,
                    typename TTypes<T>::ConstScalar l2_shrinkage,
                    typename TTypes<T>::ConstScalar lr_power,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,
                    bool multiply_linear_by_lr) {
    const Tindex N = static_cast<Tindex>(indices.dimension(0));
    if (N == 0) return Status::OK();

    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();
    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    return GpuLaunchKernel(
        SparseApplyFtrlKernel<T, Tindex, has_l2_shrinkage>, config.block_count,
        config.thread_per_block, 0, d.stream(), /*var=*/var.data(),
        /*accum=*/accum.data(),
        /*linear=*/linear.data(), /*lr=*/lr.data(), /*l1=*/l1.data(),
        /*l2=*/l2.data(), /*l2_shrinkage=*/l2_shrinkage.data(),
        /*lr_power=*/lr_power.data(), /*grad=*/grad.data(),
        /*indices=*/indices.data(), /*param_rows=*/first_dim_size,
        /*updates_size=*/grad_size,
        /*indices_size=*/indices_size,
        /*multiply_linear_by_lr=*/multiply_linear_by_lr);
  }
};

template <typename T>
struct ApplyMomentum<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    accum.device(d) = accum * momentum.reshape(single).broadcast(bcast) + grad;
    if (use_nesterov) {
      var.device(d) -= grad * lr.reshape(single).broadcast(bcast) +
                       accum * momentum.reshape(single).broadcast(bcast) *
                           lr.reshape(single).broadcast(bcast);
    } else {
      var.device(d) -= lr.reshape(single).broadcast(bcast) * accum;
    }
  }
};

template <typename T>
struct ApplyKerasMomentum<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    accum.device(d) = (accum * momentum.reshape(single).broadcast(bcast) -
                       grad * lr.reshape(single).broadcast(bcast));
    if (use_nesterov) {
      var.device(d) += (accum * momentum.reshape(single).broadcast(bcast) -
                        grad * lr.reshape(single).broadcast(bcast));
    } else {
      var.device(d) += accum;
    }
  }
};

template <typename T>
struct ApplyAdam<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    m.device(d) =
        m + (beta1.constant(one) - beta1).reshape(single).broadcast(bcast) *
                (grad - m);
    v.device(d) =
        v + (beta2.constant(one) - beta2).reshape(single).broadcast(bcast) *
                (grad.square() - v);

    if (use_nesterov) {
      var.device(d) -=
          (lr * (beta2_power.constant(one) - beta2_power).sqrt() /
           (beta1_power.constant(one) - beta1_power))
              .reshape(single)
              .broadcast(bcast) *
          (m * beta1.reshape(single).broadcast(bcast) +
           (beta1.constant(one) - beta1).reshape(single).broadcast(bcast) *
               grad) /
          (epsilon.reshape(single).broadcast(bcast) + v.sqrt());
    } else {
      var.device(d) -= (lr * (beta2_power.constant(one) - beta2_power).sqrt() /
                        (beta1_power.constant(one) - beta1_power))
                           .reshape(single)
                           .broadcast(bcast) *
                       m /
                       (epsilon.reshape(single).broadcast(bcast) + v.sqrt());
    }
  }
};

template <typename T>
struct ApplyAdamWithAmsgrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Flat vhat,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    m.device(d) =
        m + (beta1.constant(one) - beta1).reshape(single).broadcast(bcast) *
                (grad - m);
    v.device(d) =
        v + (beta2.constant(one) - beta2).reshape(single).broadcast(bcast) *
                (grad.square() - v);
    vhat.device(d) = vhat.cwiseMax(v);

    var.device(d) -= (lr * (beta2_power.constant(one) - beta2_power).sqrt() /
                      (beta1_power.constant(one) - beta1_power))
                         .reshape(single)
                         .broadcast(bcast) *
                     m /
                     (epsilon.reshape(single).broadcast(bcast) + vhat.sqrt());
  }
};

template <typename T>
struct ApplyAdaMax<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    m.device(d) =
        m + (beta1.constant(one) - beta1).reshape(single).broadcast(bcast) *
                (grad - m);
    v.device(d) =
        (beta2.reshape(single).broadcast(bcast) * v).cwiseMax(grad.abs());
    var.device(d) -=
        lr / (beta1_power.constant(one) -
                 beta1_power).reshape(single).broadcast(bcast) *
                     (m / (v + epsilon));
  }
};

template <typename T>
struct ApplyRMSProp<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    ms.device(d) =
        ms + (rho.constant(one) - rho).reshape(single).broadcast(bcast) *
                 (grad.square() - ms);
    mom.device(d) =
        mom * momentum.reshape(single).broadcast(bcast) +
        lr.reshape(single).broadcast(bcast) * grad /
            ((epsilon.reshape(single).broadcast(bcast) + ms).sqrt());
    var.device(d) -= mom;
  }
};

template <typename T>
struct ApplyCenteredRMSProp<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat mg, typename TTypes<T>::Flat ms,
                  typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    const auto one_minus_rho =
        (rho.constant(one) - rho).reshape(single).broadcast(bcast);
    ms.device(d) = ms + one_minus_rho * (grad.square() - ms);
    mg.device(d) = mg + one_minus_rho * (grad - mg);
    auto denom = (ms - mg.square()) + epsilon.reshape(single).broadcast(bcast);
    mom.device(d) = mom * momentum.reshape(single).broadcast(bcast) +
                    lr.reshape(single).broadcast(bcast) * grad / denom.sqrt();
    var.device(d) -= mom;
  }
};

template <typename T>
struct ApplyAddSign<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar alpha,
                  typename TTypes<T>::ConstScalar sign_decay,
                  typename TTypes<T>::ConstScalar beta,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    // The following is the GPU equivalent of the CPU version:
    // m.device(d) = m * beta() + grad * (static_cast<T>(1) - beta());
    const auto one = static_cast<T>(1.0);
    auto beta_bcast = beta.reshape(single).broadcast(bcast);
    auto one_minus_beta =
        (beta.constant(one) - beta).reshape(single).broadcast(bcast);
    m.device(d) = m * beta_bcast + grad * one_minus_beta;

    // The following is the GPU equivalent of the CPU version:
    // var.device(d) -= lr() * (alpha() + sign_decay() * sign_gm) * grad;
    auto sign_gm = grad.sign() * m.sign();
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto alpha_bcast = alpha.reshape(single).broadcast(bcast);
    auto sign_decay_bcast = sign_decay.reshape(single).broadcast(bcast);
    var.device(d) -=
        lr_bcast * (alpha_bcast + sign_decay_bcast * sign_gm) * grad;
  }
};

template <typename T>
struct ApplyPowerSign<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar logbase,
                  typename TTypes<T>::ConstScalar sign_decay,
                  typename TTypes<T>::ConstScalar beta,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    // The following is the GPU equivalent of the CPU version:
    // m.device(d) = m * beta() + grad * (static_cast<T>(1) - beta());
    const auto one = static_cast<T>(1.0);
    auto beta_bcast = beta.reshape(single).broadcast(bcast);
    auto one_minus_beta =
        (beta.constant(one) - beta).reshape(single).broadcast(bcast);
    m.device(d) = m * beta_bcast + grad * one_minus_beta;

    // The following is the GPU equivalent of the CPU version:
    // auto grad_scale = (logbase() * sign_decay() * sign_gm).exp();
    // var.device(d) -= lr() * grad_scale * grad;
    auto sign_gm = grad.sign() * m.sign();
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto logbase_bcast = logbase.reshape(single).broadcast(bcast);
    auto sign_decay_bcast = sign_decay.reshape(single).broadcast(bcast);
    auto grad_scale = (logbase_bcast * sign_decay_bcast * sign_gm).exp();
    var.device(d) -= lr_bcast * grad_scale * grad;
  }
};

template <typename T, typename Tindex>
__global__ __launch_bounds__(1024) void SparseApplyAdamKernel(
  T* var, T* m, T* v, const T* grad, const T* beta1_power, const T* beta2_power,
  const T* lr, const T* beta1, const T* beta2, const T* epsilon, const Tindex* indices, 
  Tindex param_rows, Tindex updates_size, Tindex indices_size) {
    Tindex col_size = updates_size / indices_size;
    const T alpha = (*lr) * sqrt(static_cast<T>(1) - *beta2_power) /
                    (static_cast<T>(1) - *beta1_power);

    GPU_1D_KERNEL_LOOP(grad_index, updates_size) {
      Tindex indices_row = grad_index / col_size;
      Tindex param_row = indices[indices_row];
      if (param_row < 0 || param_row >= param_rows) {
        // Ignore indices that are out of range
        continue;
      }

      // Index of var, m and v
      Tindex param_index = param_row*col_size + grad_index%col_size;
      const T& g = grad[grad_index];
      T& var_a = var[param_index];
      T& m_a = m[param_index];
      T& v_a = v[param_index];

      m_a += (g-m_a) * (static_cast<T>(1) - (*beta1));
      v_a += (g*g - v_a) * (static_cast<T>(1) - (*beta2));
      var_a -= (m_a*alpha) / (sqrt(v_a)+ (*epsilon));
    }
}
template <typename T, typename Tindex> 
struct SparseApplyAdam<GPUDevice, T, Tindex> {
  Status operator()(const GPUDevice& d, typename TTypes<T>::Matrix var, 
                  typename TTypes<T>::Matrix m, 
                  typename TTypes<T>::Matrix v, 
                  typename TTypes<T>::ConstMatrix grad,
                  typename TTypes<T>::ConstScalar beta1_power, 
                  typename TTypes<T>::ConstScalar beta2_power, 
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon, 
                  typename TTypes<Tindex>::ConstVec indices, 
                  const int64 inner_dim) {
    const Tindex N = static_cast<Tindex>(indices.dimension(0));
    if (N == 0) return Status::OK();

    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();

    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    return GpuLaunchKernel(SparseApplyAdamKernel<T, Tindex>, config.block_count,
                    config.thread_per_block, 0, d.stream(), var.data(), m.data(), 
                    v.data(), grad.data(), beta1_power.data(), beta2_power.data(), 
                    lr.data(), beta1.data(), beta2.data(), epsilon.data(), 
                    indices.data(), first_dim_size, grad_size, indices_size);
  }
};

template <typename T>
__global__ __launch_bounds__(1024)
void ApplyFtrlV2Kernel(T *var, T *accum, T *linear, const T *grad, const T lr, const T l1,
		       const T l2, const T l2_shrinkage, const T lr_power, const int64 grad_size) {
  GPU_1D_KERNEL_LOOP(grad_index, grad_size) {
    auto grad_with_shrinkage = grad[grad_index]  +
      static_cast<T>(2)*l2_shrinkage*var[grad_index];
    auto new_accum = accum[grad_index] + grad[grad_index]*grad[grad_index];
    // special case for which lr_power=-0.5
    if (lr == static_cast<T>(-0.5)) {
      linear[grad_index] += grad_with_shrinkage -
	(sqrt(new_accum) - sqrt(accum[grad_index])) / lr * var[grad_index];
    } else {
      linear[grad_index] += grad_with_shrinkage -
	(pow(new_accum, -lr_power) - pow(accum[grad_index], -lr_power)) / lr * var[grad_index];
    }
    T sign = linear[grad_index] < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(1);
    auto x = l1*sign - linear[grad_index];
    if (lr_power == static_cast<T>(-0.5)) {
      auto y = sqrt(new_accum) / lr + static_cast<T>(2)*l2;
      auto pre_shrink = x / y;
      var[grad_index] = (linear[grad_index] > l1 || linear[grad_index] < -l1) ?
	pre_shrink : static_cast<T>(0);
    } else {
      auto y = pow(new_accum, -lr_power) / lr + static_cast<T>(2)*l2;
      auto pre_shrink = x / y;
      var[grad_index] = (linear[grad_index] > l1 || linear[grad_index] < -l1) ?
	pre_shrink : static_cast<T>(0);
    }

    accum[grad_index] += grad[grad_index]*grad[grad_index];
  }
}
  
template <typename T>
struct ApplyFtrlV2<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar l2_shrinkage,
                  typename TTypes<T>::ConstScalar lr_power) {
    int64 grad_size = grad.size();
    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    GpuLaunchKernel(ApplyFtrlV2Kernel<T>, config.block_count, config.thread_per_block,
		    0, d.stream(), var.data(), accum.data(), linear.data(),
		    grad.data(), lr(), l1(), l2(), l2_shrinkage(), lr_power(), grad_size);
  }
};

template <typename T>
__global__ __launch_bounds__(1024)  
void ApplyFtrlKernel(T *var, T *accum, T *linear, const T *grad, const T lr, const T l1,
		     const T l2, const T lr_power, const int64 grad_size) {
  GPU_1D_KERNEL_LOOP(grad_index, grad_size) {
    auto new_accum = accum[grad_index] + grad[grad_index]*grad[grad_index];
    // special case for which lr_power=-0.5
    if (lr_power == static_cast<T>(-0.5)) {
      linear[grad_index] += grad[grad_index] -
	(sqrt(new_accum) - sqrt(accum[grad_index])) / lr * var[grad_index];
    } else {
      linear[grad_index] += grad[grad_index] -
	(pow(new_accum, -lr_power) - pow(accum[grad_index], -lr_power)) / lr * var[grad_index];
    }
    T sign = linear[grad_index] < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(1);
    auto x = l1*sign - linear[grad_index];
    if (lr_power == static_cast<T>(-0.5)) {
      auto y = sqrt(new_accum) / lr + static_cast<T>(2)*l2;
      auto pre_shrink = x / y;
      var[grad_index] = (linear[grad_index] > l1 || linear[grad_index] < -l1) ?
	pre_shrink : static_cast<T>(0);
    } else {
      auto y = pow(new_accum, -lr_power) / lr + static_cast<T>(2)*l2;
      auto pre_shrink = x / y;
      var[grad_index] = (linear[grad_index] > l1 || linear[grad_index] < -l1) ?
	pre_shrink : static_cast<T>(0);
    }
    accum[grad_index] += grad[grad_index]*grad[grad_index];
  }
}

template <typename T>
struct ApplyFtrl<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar lr_power) {
    int64 grad_size = grad.size();
    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    GpuLaunchKernel(ApplyFtrlKernel<T>, config.block_count, config.thread_per_block,
		    0, d.stream(), var.data(), accum.data(), linear.data(),
		    grad.data(), lr(), l1(), l2(), lr_power(), grad_size);
  }
};

}  // namespace functor

template struct functor::ApplyGradientDescent<GPUDevice, Eigen::half>;
template struct functor::ApplyGradientDescent<GPUDevice, float>;
template struct functor::ApplyGradientDescent<GPUDevice, double>;

template struct functor::ApplyAdagrad<GPUDevice, Eigen::half>;
template struct functor::ApplyAdagrad<GPUDevice, float>;
template struct functor::ApplyAdagrad<GPUDevice, double>;

template struct functor::ApplyAdagradV2<GPUDevice, Eigen::half>;
template struct functor::ApplyAdagradV2<GPUDevice, float>;
template struct functor::ApplyAdagradV2<GPUDevice, double>;

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T)                             \
  template struct functor::SparseApplyAdagrad<GPUDevice, T, int32,    \
                                              /*has_epsilon=*/false>; \
  template struct functor::SparseApplyAdagrad<GPUDevice, T, int64,    \
                                              /*has_epsilon=*/false>; \
  template struct functor::SparseApplyAdagrad<GPUDevice, T, int32,    \
                                              /*has_epsilon=*/true>;  \
  template struct functor::SparseApplyAdagrad<GPUDevice, T, int64,    \
                                              /*has_epsilon=*/true>
EXPLICITLY_INSTANTIATE_FUNCTOR(Eigen::half);
EXPLICITLY_INSTANTIATE_FUNCTOR(float);
EXPLICITLY_INSTANTIATE_FUNCTOR(double);
#undef EXPLICITLY_INSTANTIATE_FUNCTOR

template struct functor::ApplyProximalAdagrad<GPUDevice, Eigen::half>;
template struct functor::ApplyProximalAdagrad<GPUDevice, float>;
template struct functor::ApplyProximalAdagrad<GPUDevice, double>;

template struct functor::SparseApplyProximalAdagrad<GPUDevice, Eigen::half,
                                                    int32>;
template struct functor::SparseApplyProximalAdagrad<GPUDevice, Eigen::half,
                                                    int64>;

template struct functor::SparseApplyProximalAdagrad<GPUDevice, float, int32>;
template struct functor::SparseApplyProximalAdagrad<GPUDevice, float, int64>;
template struct functor::SparseApplyProximalAdagrad<GPUDevice, double, int32>;
template struct functor::SparseApplyProximalAdagrad<GPUDevice, double, int64>;

template struct functor::ApplyAdadelta<GPUDevice, Eigen::half>;
template struct functor::ApplyAdadelta<GPUDevice, float>;
template struct functor::ApplyAdadelta<GPUDevice, double>;

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T)                               \
  template struct functor::SparseApplyFtrl<GPUDevice, T, int32,         \
                                           /*has_l2_shrinkage=*/false>; \
  template struct functor::SparseApplyFtrl<GPUDevice, T, int64,         \
                                           /*has_l2_shrinkage=*/false>; \
  template struct functor::SparseApplyFtrl<GPUDevice, T, int32,         \
                                           /*has_l2_shrinkage=*/true>;  \
  template struct functor::SparseApplyFtrl<GPUDevice, T, int64,         \
                                           /*has_l2_shrinkage=*/true>
EXPLICITLY_INSTANTIATE_FUNCTOR(Eigen::half);
EXPLICITLY_INSTANTIATE_FUNCTOR(float);
EXPLICITLY_INSTANTIATE_FUNCTOR(double);
#undef EXPLICITLY_INSTANTIATE_FUNCTOR

template struct functor::ApplyMomentum<GPUDevice, Eigen::half>;
template struct functor::ApplyMomentum<GPUDevice, float>;
template struct functor::ApplyMomentum<GPUDevice, double>;

template struct functor::ApplyKerasMomentum<GPUDevice, Eigen::half>;
template struct functor::ApplyKerasMomentum<GPUDevice, float>;
template struct functor::ApplyKerasMomentum<GPUDevice, double>;

template struct functor::ApplyAdam<GPUDevice, Eigen::half>;
template struct functor::ApplyAdam<GPUDevice, float>;
template struct functor::ApplyAdam<GPUDevice, double>;

template struct functor::ApplyAdamWithAmsgrad<GPUDevice, Eigen::half>;
template struct functor::ApplyAdamWithAmsgrad<GPUDevice, float>;
template struct functor::ApplyAdamWithAmsgrad<GPUDevice, double>;

template struct functor::ApplyAdaMax<GPUDevice, Eigen::half>;
template struct functor::ApplyAdaMax<GPUDevice, float>;
template struct functor::ApplyAdaMax<GPUDevice, double>;

template struct functor::ApplyRMSProp<GPUDevice, Eigen::half>;
template struct functor::ApplyRMSProp<GPUDevice, float>;
template struct functor::ApplyRMSProp<GPUDevice, double>;

template struct functor::ApplyCenteredRMSProp<GPUDevice, Eigen::half>;
template struct functor::ApplyCenteredRMSProp<GPUDevice, float>;
template struct functor::ApplyCenteredRMSProp<GPUDevice, double>;

template struct functor::ApplyAddSign<GPUDevice, Eigen::half>;
template struct functor::ApplyAddSign<GPUDevice, float>;
template struct functor::ApplyAddSign<GPUDevice, double>;

template struct functor::ApplyPowerSign<GPUDevice, Eigen::half>;
template struct functor::ApplyPowerSign<GPUDevice, float>;
template struct functor::ApplyPowerSign<GPUDevice, double>;

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T)                          \
  template struct functor::SparseApplyAdam<GPUDevice, T, int32>;  \
  template struct functor::SparseApplyAdam<GPUDevice, T, int64>;
EXPLICITLY_INSTANTIATE_FUNCTOR(Eigen::half);
EXPLICITLY_INSTANTIATE_FUNCTOR(float);
EXPLICITLY_INSTANTIATE_FUNCTOR(double);
#undef EXPLICITLY_INSTANTIATE_FUNCTOR

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T)		\
  template struct functor::ApplyFtrl<GPUDevice, T>;	\
  template struct functor::ApplyFtrlV2<GPUDevice, T>;
  EXPLICITLY_INSTANTIATE_FUNCTOR(Eigen::half);
  EXPLICITLY_INSTANTIATE_FUNCTOR(float);
  EXPLICITLY_INSTANTIATE_FUNCTOR(double);
#undef EXPLICITLY_INSTANTIATE_FUNCTOR
  
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

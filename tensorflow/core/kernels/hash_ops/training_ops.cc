/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>

#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/kernels/training_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/hash_table/tensible_variable.h"
#include "tensorflow/core/framework/hash_table/hash_table.h"
#include "tensorflow/core/framework/hash_table/status_collector.h"

namespace tensorflow {

namespace functor {

namespace {
Eigen::DefaultDevice simple_device_;
}

template <typename T>
struct SimpleApplyGradientDescent {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  T lr,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    var.device(simple_device_) -= grad * lr;
  }
};

template <typename T>
struct SimpleApplyAdadelta {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat accum,
                  typename TTypes<T>::UnalignedFlat accum_update,
                  T lr,
                  T rho,
                  T epsilon,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    accum.device(simple_device_) =
        accum * rho + grad.square() * (T(1) - rho);
    const auto update =
        (accum_update + epsilon).sqrt() * (accum + epsilon).rsqrt() * grad;
    var.device(simple_device_) -= update * lr;
    accum_update.device(simple_device_) =
        accum_update * rho + update.square() * (T(1) - rho);
  }
};

template <typename T>
struct SimpleApplyProximalGradientDescent {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  T lr,
                  T l1,
                  T l2,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    // Note that here is Fobos update, for details please refer:
    // http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf
    // TODO(xbing): merge the logic for ProximalGradientDescent and
    // ProximalAdagrad.
    auto prox_var = var;
    // compute v = w - lr * grad.
    prox_var.device(simple_device_) -= grad * lr;
    if (l1 > T(0)) {
      // compute sign(v) * max(|v| - lr * l1, 0)
      var.device(simple_device_) =
          prox_var.sign() *
          (prox_var.abs() - var.constant(lr * l1)).cwiseMax(T(0.0)) /
          (var.constant(T(1.0)) + var.constant(l2 * lr));
    } else {
      var.device(simple_device_) =
          prox_var / (var.constant(T(1.0)) + var.constant(l2 * lr));
    }
  }
};

template <typename T>
struct SimpleApplyAdagradDA {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat gradient_accum,
                  typename TTypes<T>::UnalignedFlat gradient_squared_accum,
                  T lr, int64 global_step,
                  T l1,
                  T l2,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    // Accumulate gradient, and gradient_squared
    gradient_accum.device(simple_device_) += grad;
    gradient_squared_accum.device(simple_device_) += grad.square();

    // AdagradDA update:
    // Let g to be gradient accumulator, gg to be gradient squared accumulator,
    // T be the global step, lr is the learning rate, and k the initial
    // gradient squared accumulator value.
    // w = \dfrac{sign(-g)*lr*|g - l1*T|_{+}}{l2*T*lr + \sqrt{k+gg})}
    if (l1 > T(0)) {
      var.device(simple_device_) =
          lr * var.constant(T(-1.0)) * gradient_accum.sign() *
          (gradient_accum.abs() -
           var.constant(static_cast<T>(global_step)) * var.constant(l1))
              .cwiseMax(T(0.0)) /
          (var.constant(l2) *
               var.constant(static_cast<T>(global_step) * lr) +
           gradient_squared_accum.sqrt());
    } else {
      var.device(simple_device_) =
          lr * gradient_accum * var.constant(T(-1.0)) /
          (var.constant(l2) *
               var.constant(static_cast<T>(global_step) * lr) +
           gradient_squared_accum.sqrt());
    }
  }
};

template <typename T>
struct SimpleApplyAdagrad {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat accum,
                  T lr,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    accum.device(simple_device_) += grad.square();
    var.device(simple_device_) -= grad * lr * accum.rsqrt();
  }
};

template <typename T>
struct SimpleApplyProximalAdagrad {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat accum,
                  T lr,
                  T l1,
                  T l2,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    // Fobos update per paper with Adagrad learning rate.
    accum.device(simple_device_) += grad.square();
    // Adagrad learning rate.
    auto learning_rate = accum.constant(lr) * accum.rsqrt();
    auto prox_var = var;
    // compute v = w - lr * grad.
    prox_var.device(simple_device_) -= grad * learning_rate;
    if (l1 > T(0)) {
      // compute sign(v) * max(|v| - lr * l1, 0)
      var.device(simple_device_) = prox_var.sign() *
                      (prox_var.abs() - learning_rate * prox_var.constant(l1))
                          .cwiseMax(T(0.0)) /
                      (var.constant(T(1.0)) + var.constant(l2) * learning_rate);
    } else {
      var.device(simple_device_) =
          prox_var / (var.constant(T(1.0)) + var.constant(l2) * learning_rate);
    }
  }
};

template <typename T>
struct SimpleApplyFtrlV2 {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat accum,
                  typename TTypes<T>::UnalignedFlat linear,
                  typename TTypes<T>::UnalignedConstFlat grad,
                  T lr,
                  T l1,
                  T l2,
                  T l2_shrinkage,
                  T lr_power) {
    auto grad_with_shrinkage = grad + static_cast<T>(2) * l2_shrinkage * var;
    auto new_accum = accum + grad_with_shrinkage.square();
    // special case for which lr_power=-0.5.
    if (lr_power == static_cast<T>(-0.5)) {
      linear.device(simple_device_) +=
          grad_with_shrinkage - (new_accum.sqrt() - accum.sqrt()) / lr * var;
    } else {
      linear.device(simple_device_) +=
          grad_with_shrinkage -
          (new_accum.pow(-lr_power) - accum.pow(-lr_power)) / lr * var;
    }
    auto x = (linear.constant(l1) * linear.sign() - linear);
    if (lr_power == static_cast<T>(-0.5)) {
      auto y = new_accum.sqrt() / new_accum.constant(lr) +
               linear.constant(static_cast<T>(2) * l2);
      auto pre_shrink = x / y;
      var.device(simple_device_) = (linear.abs() > linear.constant(l1))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));

    } else {
      auto y = new_accum.pow(-lr_power) / new_accum.constant(lr) +
               linear.constant(static_cast<T>(2) * l2);
      auto pre_shrink = x / y;
      var.device(simple_device_) = (linear.abs() > linear.constant(l1))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));
    }
    accum.device(simple_device_) += grad_with_shrinkage.square();
  }
};

template <typename T>
struct SimpleApplyFtrl {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat accum,
                  typename TTypes<T>::UnalignedFlat linear,
                  typename TTypes<T>::UnalignedConstFlat grad,
                  T lr,
                  T l1,
                  T l2,
                  T lr_power) {
    auto new_accum = accum + grad.square();
    // special case for which lr_power=-0.5.
    if (lr_power == static_cast<T>(-0.5)) {
      linear.device(simple_device_) += grad - (new_accum.sqrt() - accum.sqrt()) / lr * var;
    } else {
      linear.device(simple_device_) +=
          grad -
          (new_accum.pow(-lr_power) - accum.pow(-lr_power)) / lr * var;
    }
    auto x = (linear.constant(l1) * linear.sign() - linear);
    if (lr_power == static_cast<T>(-0.5)) {
      auto y = new_accum.sqrt() / new_accum.constant(lr) +
               linear.constant(static_cast<T>(2) * l2);
      auto pre_shrink = x / y;
      var.device(simple_device_) = (linear.abs() > linear.constant(l1))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));

    } else {
      auto y = new_accum.pow(-lr_power) / new_accum.constant(lr) +
               linear.constant(static_cast<T>(2) * l2);
      auto pre_shrink = x / y;
      var.device(simple_device_) = (linear.abs() > linear.constant(l1))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));
    }
    accum.device(simple_device_) += grad.square();
  }
};

template <typename T>
struct SimpleApplyMomentumNesterov {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat accum,
                  T lr,
                  typename TTypes<T>::UnalignedConstFlat grad,
                  T momentum) {
    accum.device(simple_device_) = accum * momentum + grad;
    var.device(simple_device_) -= grad * lr + accum * momentum * lr;
  }
};

template <typename T>
struct SimpleApplyMomentumNoNesterov {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat accum,
                  T lr,
                  typename TTypes<T>::UnalignedConstFlat grad,
                  T momentum) {
    accum.device(simple_device_) = accum * momentum + grad;
    var.device(simple_device_) -= accum * lr;
  }
};

template <typename T>
struct SimpleApplyAdamNesterov {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat m,
                  typename TTypes<T>::UnalignedFlat v,
                  T alpha,
                  T lr,
                  T beta1,
                  T beta2,
                  T epsilon,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    (void)lr;
    m.device(simple_device_) += (grad - m) * (T(1) - beta1);
    v.device(simple_device_) += (grad.square() - v) * (T(1) - beta2);
    var.device(simple_device_) -= ((grad * (T(1) - beta1) + beta1 * m) * alpha) /
                     (v.sqrt() + epsilon);
  }
};

template <typename T>
struct SimpleApplyAdamNoNesterov {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat m,
                  typename TTypes<T>::UnalignedFlat v,
                  T alpha,
                  T lr,
                  T beta1,
                  T beta2,
                  T epsilon,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    (void)lr;
    m.device(simple_device_) += (grad - m) * (T(1) - beta1);
    v.device(simple_device_) += (grad.square() - v) * (T(1) - beta2);
    var.device(simple_device_) -= (m * alpha) / (v.sqrt() + epsilon);
  }
};

template <typename T>
struct SimpleApplyRMSProp {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat ms,
                  typename TTypes<T>::UnalignedFlat mom,
                  T lr,
                  T rho,
                  T momentum,
                  T epsilon,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    ms.device(simple_device_) += (grad.square() - ms) * (static_cast<T>(1) - rho);
    mom.device(simple_device_) =
        mom * momentum + (grad * lr) / ((ms + epsilon).sqrt());
    var.device(simple_device_) -= mom;
  }
};

template <typename T>
struct SimpleApplyCenteredRMSProp {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat mg,
                  typename TTypes<T>::UnalignedFlat ms,
                  typename TTypes<T>::UnalignedFlat mom,
                  T lr,
                  T rho,
                  T momentum,
                  T epsilon,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    ms.device(simple_device_) += (grad.square() - ms) * (static_cast<T>(1) - rho);
    mg.device(simple_device_) += (grad - mg) * (static_cast<T>(1) - rho);
    auto denom = (ms - mg.square()) + epsilon;
    mom.device(simple_device_) = mom * momentum + (grad * lr) / denom.sqrt();
    var.device(simple_device_) -= mom;
  }
};

template <typename T>
struct SimpleApplyAddSign {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat m,
                  T lr,
                  T alpha,
                  T sign_decay,
                  T beta,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    m.device(simple_device_) = m * beta + grad * (static_cast<T>(1) - beta);
    auto sign_gm = grad.sign() * m.sign();
    var.device(simple_device_) -= lr * (alpha + sign_decay * sign_gm) * grad;
  }
};

template <typename T>
struct SimpleApplyPowerSign {
  void operator()(typename TTypes<T>::UnalignedFlat var,
                  typename TTypes<T>::UnalignedFlat m,
                  T lr,
                  T logbase,
                  T sign_decay,
                  T beta,
                  typename TTypes<T>::UnalignedConstFlat grad) {
    m.device(simple_device_) = m * beta + grad * (static_cast<T>(1) - beta);
    auto sign_gm = grad.sign() * m.sign();
    auto grad_scale = (logbase * sign_decay * sign_gm).exp();
    var.device(simple_device_) -= lr * grad_scale * grad;
  }
};
}  // namespace functor

namespace {

struct TensibleVariableHolder {
 public:
  TensibleVariableHolder(): lock_(false) {}
  TensibleVariableHolder(TensibleVariableHolder&& rhs) {
    resources_.swap(rhs.resources_);
    locker_.swap(rhs.locker_);
    lock_ = rhs.lock_;
  }
  TensibleVariableHolder(const TensibleVariableHolder& rhs) = delete;
  Status Init(OpKernelContext* ctx, std::vector<int> ids,  bool lock) {
    lock_ = lock;
    for (auto id : ids) {
      TensibleVariableResource* resource;
      TF_RETURN_IF_ERROR(
          LookupResource(ctx, HandleFromInput(ctx, id), &resource));
      resources_.push_back(resource);
      if (resource->Internal() == nullptr) {
        return errors::FailedPrecondition("TensibleVariable is not initialized ", id);
      }
    }

    locker_ = resources_;
    std::sort(locker_.begin(), locker_.end());
    for (auto locker: locker_) {
      if (lock_) {
        locker->Internal()->LockUpdate();        
      }

      locker->Internal()->GetRWLock()->lock_shared();
    }

    return Status::OK();
  }
  ~TensibleVariableHolder() {
    for (int i = locker_.size() - 1; i >= 0; i--) {
      if (lock_) {
        locker_[i]->Internal()->UnlockUpdate();
      }

      locker_[i]->Internal()->GetRWLock()->unlock_shared();
    }
    for (auto resource : resources_) {
      resource->Unref();
    }
  }
  TensibleVariable* Get(int id) {
    return resources_[id]->Internal();
  }

 private:
  bool lock_;
  std::vector<TensibleVariableResource*> resources_;
  std::vector<TensibleVariableResource*> locker_;
};

Status ValidateUpdaterShapeAndDtype(
    std::vector<TensibleVariable*> vars, std::vector<DataType> dtypes,
    const Tensor& id, const Tensor& grad,
    int64* slice, int64* size, int64* max_id) {
  int64 oslice = 1;
  std::vector<int64> var_dims;
  const TensorShape& var_shape = vars[0]->shape();
  for (int i = 1; i < var_shape.dims(); i++) {
    var_dims.push_back(var_shape.dim_size(i));
    oslice *= var_shape.dim_size(i);
  }
  for (size_t i = 1; i < vars.size(); i++) {
    const TensorShape& var_shape2 = vars[i]->shape();
    if (static_cast<int64>(var_dims.size()) != var_shape2.dims() - 1) {
      return errors::InvalidArgument(
          "var ", i, " shape mismatch ", vars[0]->shape().DebugString(),
          " vs ", vars[i]->shape().DebugString());
    }
    for (size_t j = 0; j < var_dims.size(); j++) {
      if (var_dims[j] != var_shape2.dim_size(j + 1)) {
        return errors::InvalidArgument(
            "var ", i, " shape mismatch ", vars[0]->shape().DebugString(),
            " vs ", vars[i]->shape().DebugString());
      }
    }
  }
  std::vector<int64> id_dims;
  const TensorShape& id_shape = id.shape();
  int64 osize = 1;
  for (int i = 0; i < id_shape.dims(); i++) {
    id_dims.push_back(id_shape.dim_size(i));
    osize *= id_shape.dim_size(i);
  }
  for (auto dim : var_dims) {
    id_dims.push_back(dim);
  }
  TensorShape grad_shape(id_dims);
  if (grad.shape() != grad_shape) {
    return errors::InvalidArgument(
        "grad shape mismatch ", grad.shape().DebugString(),
        " vs ", grad_shape.DebugString());
  }
  int64 omax_id = 0;
  for (auto var : vars) {
    omax_id = std::max(omax_id, var->Size());
  }
  for (size_t i = 0; i < vars.size(); i++) {
    if (vars[i]->dtype() != dtypes[i]) {
      return errors::InvalidArgument(
          "var ", i, " dtype mismatch ", vars[i]->dtype(),
          " vs ", dtypes[i]);
    }
  }
  *slice = oslice;
  *size = osize;
  *max_id = omax_id;
  return Status::OK();
}

template<typename T>
Status GetScalar(
    OpKernelContext* ctx, int id, const char* name,
    T* result) {
  const Tensor& tensor = ctx->input(id);
  if (!TensorShapeUtils::IsScalar(tensor.shape())) {
      return errors::InvalidArgument(name, " is not a scalar: ",
                                     tensor.shape().DebugString());
  }
  *result = tensor.scalar<T>()();
  return Status::OK();
}

template<typename T>
T* GetData(const Tensor& tensor) {
  return reinterpret_cast<T*>(const_cast<char*>(
        tensor.tensor_data().data()));
}

static constexpr int kIdBlockSize = 4096;

}  // namespace

template <typename T>
class TensibleVariableApplyAdagrad : public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyAdagrad(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T lr;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 2, "lr", &lr), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0, 1}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* accum = holder.Get(1);
    const Tensor& grad = ctx->input(3);
    const Tensor& ids = ctx->input(4);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, accum}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));
    auto fn = [slice, max_id, pid, pgrad, var, accum, lr]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyAdagrad<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(
              var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_accum(
              accum->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
          apply(flat_var, flat_accum, lr, flat_grad);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

template <typename T>
class TensibleVariableApplyMomentum : public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyMomentum(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T lr, momentum;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 2, "lr", &lr), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 5, "momentum", &momentum), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0, 1}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* accum = holder.Get(1);
    const Tensor& grad = ctx->input(3);
    const Tensor& ids = ctx->input(4);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, accum}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));
    if (use_nesterov_) {
      auto fn = [slice, max_id, pid, pgrad, var, accum, lr, momentum]
          (int64 offset, int64 size) -> Status {
        typename functor::SimpleApplyMomentumNesterov<T> apply;
        Eigen::array<Eigen::DenseIndex, 1> slice_shape;
        slice_shape[0] = slice;
        int64* slice_id = pid + offset;
        T* slice_grad = pgrad + offset * slice;
        for (int64 i = 0; i < size; i++) {
          if (*slice_id != HashTable::kNotAdmitted) {
            if (*slice_id < 0 || *slice_id >= max_id) {
              return errors::InvalidArgument("Id Out of range ", *slice_id);
            }
            typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedFlat flat_accum(accum->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
            apply(flat_var, flat_accum, lr, flat_grad, momentum);
          }
          slice_id++;
          slice_grad += slice;
        }
        return Status::OK();
      };
      auto done_fn = [done, ctx, pholder](Status st) {
        delete pholder;
        OP_REQUIRES_OK_ASYNC(ctx, st, done);
        done();
      };
      ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
    } else {
      auto fn = [slice, max_id, pid, pgrad, var, accum, lr, momentum]
          (int64 offset, int64 size) -> Status {
        typename functor::SimpleApplyMomentumNoNesterov<T> apply;
        Eigen::array<Eigen::DenseIndex, 1> slice_shape;
        slice_shape[0] = slice;
        int64* slice_id = pid + offset;
        T* slice_grad = pgrad + offset * slice;
        for (int64 i = 0; i < size; i++) {
          if (*slice_id != HashTable::kNotAdmitted) {
            if (*slice_id < 0 || *slice_id >= max_id) {
              return errors::InvalidArgument("Id Out of range ", *slice_id);
            }
            typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedFlat flat_accum(accum->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
            apply(flat_var, flat_accum, lr, flat_grad, momentum);
          }
          slice_id++;
          slice_grad += slice;
        }
        return Status::OK();
      };
      auto done_fn = [done, ctx, pholder](Status st) {
        delete pholder;
        OP_REQUIRES_OK_ASYNC(ctx, st, done);
        done();
      };
      ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
    }
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

template <typename T>
class TensibleVariableApplyGradientDescent : public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyGradientDescent(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T alpha;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 1, "alpha", &alpha), done);
    const Tensor& grad = ctx->input(2);
    const Tensor& ids = ctx->input(3);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));

    auto fn = [slice, max_id, pid, pgrad, var, alpha]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyGradientDescent<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(
              var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(
              slice_grad, slice_shape);
          apply(flat_var, alpha, flat_grad);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class TensibleVariableApplyProximalGradientDescent : public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyProximalGradientDescent(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T alpha, l1, l2;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 1, "alpha", &alpha), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 2, "l1", &l1), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 3, "l2", &l2), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    const Tensor& grad = ctx->input(4);
    const Tensor& ids = ctx->input(5);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));

    auto fn = [slice, max_id, pid, pgrad, var, alpha, l1, l2]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyProximalGradientDescent<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
          apply(flat_var, alpha, l1, l2, flat_grad);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class TensibleVariableApplyAdadelta: public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyAdadelta(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T lr, rho, epsilon;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 3, "lr", &lr), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 4, "rho", &rho), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 5, "epsilon", &epsilon), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0, 1, 2}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* accum = holder.Get(1);
    TensibleVariable* accum_update = holder.Get(2);
    const Tensor& grad = ctx->input(6);
    const Tensor& ids = ctx->input(7);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, accum, accum_update}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));

    auto fn = [slice, max_id, pid, pgrad, var, accum, accum_update, lr, rho, epsilon]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyAdadelta<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_accum(accum->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_accum_update(accum_update->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
          apply(flat_var, flat_accum, flat_accum_update, lr, rho, epsilon, flat_grad);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class TensibleVariableApplyAdagradDA: public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyAdagradDA(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T lr, l1, l2;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 5, "lr", &lr), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 6, "l1", &l1), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 7, "l2", &l2), done);
    int64 global_step;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<int64>(ctx, 8, "global_step", &global_step), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0, 1, 2}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* accum = holder.Get(1);
    TensibleVariable* squared_accum = holder.Get(2);
    const Tensor& grad = ctx->input(3);
    const Tensor& ids = ctx->input(4);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, accum, squared_accum}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));

    auto fn = [slice, max_id, pid, pgrad, var, accum, squared_accum, lr, l1, l2, global_step]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyAdagradDA<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_accum(accum->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_squared_accum(squared_accum->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
          apply(flat_var, flat_accum, flat_squared_accum, lr, global_step, l1, l2, flat_grad);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class TensibleVariableApplyProximalAdagrad: public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyProximalAdagrad(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T lr, l1, l2;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 2, "lr", &lr), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 3, "l1", &l1), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 4, "l2", &l2), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0, 1}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* accum = holder.Get(1);
    const Tensor& grad = ctx->input(5);
    const Tensor& ids = ctx->input(6);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, accum}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));

    auto fn = [slice, max_id, pid, pgrad, var, accum, lr, l1, l2]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyProximalAdagrad<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_accum(accum->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
          apply(flat_var, flat_accum, lr, l1, l2, flat_grad);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class TensibleVariableApplyFtrl : public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyFtrl(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T lr, l1, l2, lr_power;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 5, "lr", &lr), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 6, "l1", &l1), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 7, "l2", &l2), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 8, "lr_power", &lr_power), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0, 1, 2}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* accum = holder.Get(1);
    TensibleVariable* linear = holder.Get(2);
    const Tensor& grad = ctx->input(3);
    const Tensor& ids = ctx->input(4);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, accum, linear}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));

    auto fn = [slice, max_id, pid, pgrad, var, accum, linear, lr, l1, l2, lr_power]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyFtrl<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_accum(accum->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_linear(linear->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
          apply(flat_var, flat_accum, flat_linear, flat_grad, lr, l1, l2, lr_power);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class TensibleVariableApplyFtrlV2 : public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyFtrlV2(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T lr, l1, l2, l2_shrinkage, lr_power;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 5, "lr", &lr), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 6, "l1", &l1), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 7, "l2", &l2), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 8, "l2_shrinkage", &l2_shrinkage), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 9, "lr_power", &lr_power), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0, 1, 2}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* accum = holder.Get(1);
    TensibleVariable* linear = holder.Get(2);
    const Tensor& grad = ctx->input(3);
    const Tensor& ids = ctx->input(4);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, accum, linear}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));

    auto fn = [slice, max_id, pid, pgrad, var, accum, linear, lr, l1, l2, l2_shrinkage, lr_power]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyFtrlV2<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_accum(accum->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_linear(linear->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
          apply(flat_var, flat_accum, flat_linear, flat_grad, lr, l1, l2, l2_shrinkage, lr_power);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class TensibleVariableApplyRMSProp: public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyRMSProp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T lr, rho, momentum, epsilon;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 3, "lr", &lr), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 4, "rho", &rho), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 5, "momentum", &momentum), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 6, "epsilon", &epsilon), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0, 1, 2}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* ms = holder.Get(1);
    TensibleVariable* mom = holder.Get(2);
    const Tensor& grad = ctx->input(7);
    const Tensor& ids = ctx->input(8);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, ms, mom}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));

    auto fn = [slice, max_id, pid, pgrad, var, ms, mom, lr, rho, momentum, epsilon]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyRMSProp<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_ms(ms->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_mom(mom->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
          apply(flat_var, flat_ms, flat_mom, lr, rho, momentum, epsilon, flat_grad);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class TensibleVariableApplyCenteredRMSProp: public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyCenteredRMSProp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T lr, rho, momentum, epsilon;
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 4, "lr", &lr), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 5, "rho", &rho), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 6, "momentum", &momentum), done);
    OP_REQUIRES_OK_ASYNC(ctx, GetScalar<T>(ctx, 7, "epsilon", &epsilon), done);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx, holder.Init(ctx, {0, 1, 2, 3}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* mg = holder.Get(1);
    TensibleVariable* ms = holder.Get(2);
    TensibleVariable* mom = holder.Get(3);
    const Tensor& grad = ctx->input(8);
    const Tensor& ids = ctx->input(9);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, mg, ms, mom}, {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));

    auto fn = [slice, max_id, pid, pgrad, var, mg, ms, mom, lr, rho, momentum, epsilon]
        (int64 offset, int64 size) -> Status {
      typename functor::SimpleApplyCenteredRMSProp<T> apply;
      Eigen::array<Eigen::DenseIndex, 1> slice_shape;
      slice_shape[0] = slice;
      int64* slice_id = pid + offset;
      T* slice_grad = pgrad + offset * slice;
      for (int64 i = 0; i < size; i++) {
        if (*slice_id != HashTable::kNotAdmitted) {
          if (*slice_id < 0 || *slice_id >= max_id) {
            return errors::InvalidArgument("Id Out of range ", *slice_id);
          }
          typename TTypes<T>::UnalignedFlat flat_var(var->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_mg(mg->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_ms(ms->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedFlat flat_mom(mom->GetSlice<T>(*slice_id), slice_shape);
          typename TTypes<T>::UnalignedConstFlat flat_grad(slice_grad, slice_shape);
          apply(flat_var, flat_mg, flat_ms, flat_mom, lr, rho, momentum, epsilon, flat_grad);
        }
        slice_id++;
        slice_grad += slice;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx, pholder](Status st) {
      delete pholder;
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class TensibleVariableApplyAdam : public AsyncOpKernel {
 public:
  explicit TensibleVariableApplyAdam(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    T beta1_power, beta2_power, lr, beta1, beta2, epsilon;
    OP_REQUIRES_OK_ASYNC(ctx,
        GetScalar<T>(ctx, 3, "beta1_power", &beta1_power), done);
    OP_REQUIRES_OK_ASYNC(ctx,
        GetScalar<T>(ctx, 4, "beta2_power", &beta2_power), done);
    OP_REQUIRES_OK_ASYNC(ctx,
        GetScalar<T>(ctx, 5, "lr", &lr), done);
    OP_REQUIRES_OK_ASYNC(ctx,
        GetScalar<T>(ctx, 6, "beta1", &beta1), done);
    OP_REQUIRES_OK_ASYNC(ctx,
        GetScalar<T>(ctx, 7, "beta2", &beta2), done);
    OP_REQUIRES_OK_ASYNC(ctx,
        GetScalar<T>(ctx, 8, "epsilon", &epsilon), done);

    T alpha = lr * Eigen::numext::sqrt(T(1) - beta2_power) /
        (T(1) - beta1_power);

    TensibleVariableHolder holder;
    OP_REQUIRES_OK_ASYNC(ctx,
        holder.Init(ctx, {0, 1, 2}, use_exclusive_lock_), done);
    TensibleVariable* var = holder.Get(0);
    TensibleVariable* m = holder.Get(1);
    TensibleVariable* v = holder.Get(2);
    const Tensor& grad = ctx->input(9);
    const Tensor& ids = ctx->input(10);

    int64 slice, size, max_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ValidateUpdaterShapeAndDtype(
            {var, v, m},
            {DataTypeToEnum<T>::value, DataTypeToEnum<T>::value,
              DataTypeToEnum<T>::value},
            ids, grad, &slice, &size, &max_id), done);
    int64* pid = GetData<int64>(ids);
    T* pgrad = GetData<T>(grad);
    auto pholder = new TensibleVariableHolder(std::move(holder));
    if (use_nesterov_) {
      auto fn = [slice, max_id, pid, pgrad, var, v, m,
                 lr, epsilon, beta1, beta2, alpha]
          (int64 offset, int64 size) -> Status {
        typename functor::SimpleApplyAdamNesterov<T> apply;
        Eigen::array<Eigen::DenseIndex, 1> slice_shape;
        slice_shape[0] = slice;
        int64* slice_id = pid + offset;
        T* slice_grad = pgrad + offset * slice;
        for (int64 i = 0; i < size; i++) {
          if (*slice_id != HashTable::kNotAdmitted) {
            if (*slice_id < 0 || *slice_id >= max_id) {
              return errors::InvalidArgument("Id Out of range ", *slice_id);
            }
            typename TTypes<T>::UnalignedFlat flat_var(
                var->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedFlat flat_v(
                v->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedFlat flat_m(
                m->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedConstFlat flat_grad(
                slice_grad, slice_shape);
            apply(flat_var, flat_m, flat_v, alpha, lr, beta1, beta2, epsilon,
                flat_grad);
          }
          slice_id++;
          slice_grad += slice;
        }
        return Status::OK();
      };
      auto done_fn = [done, ctx, pholder](Status st) {
        delete pholder;
        OP_REQUIRES_OK_ASYNC(ctx, st, done);
        done();
      };
      ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
    } else {
      auto fn = [slice, max_id, pid, pgrad, var, v, m, lr,
                 epsilon, beta1, beta2, alpha]
          (int64 offset, int64 size) -> Status {
        typename functor::SimpleApplyAdamNoNesterov<T> apply;
        Eigen::array<Eigen::DenseIndex, 1> slice_shape;
        slice_shape[0] = slice;
        int64* slice_id = pid + offset;
        T* slice_grad = pgrad + offset * slice;
        for (int64 i = 0; i < size; i++) {
          if (*slice_id != HashTable::kNotAdmitted) {
            if (*slice_id < 0 || *slice_id >= max_id) {
              return errors::InvalidArgument("Id Out of range ", *slice_id);
            }
            typename TTypes<T>::UnalignedFlat flat_var(
                var->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedFlat flat_v(
                v->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedFlat flat_m(
                m->GetSlice<T>(*slice_id), slice_shape);
            typename TTypes<T>::UnalignedConstFlat flat_grad(
                slice_grad, slice_shape);
            apply(
                flat_var, flat_m, flat_v, alpha, lr,
                beta1, beta2, epsilon, flat_grad);
          }
          slice_id++;
          slice_grad += slice;
        }
        return Status::OK();
      };
      auto done_fn = [done, ctx, pholder](Status st) {
        delete pholder;
        OP_REQUIRES_OK_ASYNC(ctx, st, done);
        done();
      };
      ParrellRun(size, kIdBlockSize, *ctx->runner(), fn, done_fn);
    }
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_UPDATER(updater, type)                           \
  REGISTER_KERNEL_BUILDER(Name(#updater)                          \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<type>("T"),         \
                          updater<type>);

#define REGISTER_ALL_TYPE_UPDATER(updater)                        \
  REGISTER_UPDATER(updater, float)                                \
  REGISTER_UPDATER(updater, double) //                               \
  // REGISTER_UPDATER(updater, Eigen::half)                          \
  // REGISTER_UPDATER(updater, bfloat16)

REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyGradientDescent)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyProximalGradientDescent)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyAdadelta)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyAdagrad)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyAdagradDA)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyProximalAdagrad)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyFtrl)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyFtrlV2)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyMomentum)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyAdam)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyRMSProp)
REGISTER_ALL_TYPE_UPDATER(TensibleVariableApplyCenteredRMSProp)

#undef REGISTER_ALL_TYPE_UPDATER
#undef REGISTER_UPDATER


}  // namespace tensorflow

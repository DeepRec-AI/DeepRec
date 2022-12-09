#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)

#define EIGEN_USE_THREADS

#include "compile_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Dice fusion op for inference, rvar and mean is frozen as 1-d constant.
template <typename T>
class DiceOp : public OpKernel {
 public:
  explicit DiceOp(OpKernelConstruction* context) : OpKernel(context) {}

  ~DiceOp() {}

  // Dice(x) = [gamma + (1-gamma)p(x)]*x
  // p(x) = sigmoid[(x-mean)*rvar]
  // Let t = (x-mean) * rvar
  // p(x) = 1/(1 + exp(-t))
  // Dice(x) = [gamma + (1-gamma)/(1 + exp(-t)) ] * x
  // Dice(x) = [1 + gamma * exp(-t)] / [1 + exp(-t)] * x
  // Dice(x) = [gamma + exp(t)] / [1 + exp(t)] * x
  void Compute(OpKernelContext* context) override {
    // Grab the input
    const Tensor* x_tensor = &context->input(0);
    const Tensor* mean_tensor = &context->input(1);
    const Tensor* rvar_tensor = &context->input(2);
    const Tensor* gamma_tensor = &context->input(3);

    const T* x = x_tensor->flat<T>().data();
    const T* mean = mean_tensor->flat<T>().data();
    const T* rvar = rvar_tensor->flat<T>().data();
    const T* gamma = gamma_tensor->flat<T>().data();

    int64 cols = x_tensor->dim_size(x_tensor->dims() - 1);
    int64 rows = 1;
    for (int64 i = 0; i < x_tensor->dims() - 1; ++i) {
      rows *= x_tensor->dim_size(i);
    }

    int64 gamma_size = 1;
    for (int64 i = 0; i < gamma_tensor->dims(); ++i) {
      gamma_size *= gamma_tensor->dim_size(i);
    }

    // To check the input
    OP_REQUIRES(context, (mean_tensor->dims() == 1),
                errors::InvalidArgument("dims(mean) != 1"));
    OP_REQUIRES(context, (rvar_tensor->dims() == 1),
                errors::InvalidArgument("dims(rvar) != 1"));

    OP_REQUIRES(
        context, (mean_tensor->dim_size(0) == cols),
        errors::InvalidArgument("size(mean) != last_dim_size_of_input"));
    OP_REQUIRES(
        context, (rvar_tensor->dim_size(0) == cols),
        errors::InvalidArgument("size(rvar) != last_dim_size_of_input"));
    OP_REQUIRES(
        context, (gamma_size == cols),
        errors::InvalidArgument("size(gamma) != last_dim_size_of_input"));

    // Create output tensors
    Tensor* y_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, x_tensor->shape(), &y_tensor));
    T* y = y_tensor->flat<T>().data();

    // Do it
    // Let every thread compute 16 rows to avoid false sharing
    const int64 total_unit = (rows + 15) / 16;
    const int64 unit_cost =
        16 * cols * 50;  // assume every element consumes 50 cycles

    auto& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool* thread_pool = worker_threads.workers;

    thread_pool->ParallelFor(
        total_unit, unit_cost, [&](int64 begin_unit, int64 end_unit) {
          auto begin_row = begin_unit * 16;
          auto end_row = end_unit * 16;
          if (end_row > rows) {
            end_row = rows;
          }
          dice(x, mean, rvar, gamma, y, cols, begin_row, end_row);
        });
  }

 private:
  void dice(const float* x, const float* mean, const float* rvar,
            const float* gamma, float* y, int64 cols, int64 begin_row,
            int64 end_row) {
    int i = begin_row;
    for (; i + 3 < end_row; i += 4) {
      dice_avx3<4>(x, mean, rvar, gamma, y, cols, i);
    }
    for (; i < end_row; ++i) {
      dice_avx3<1>(x, mean, rvar, gamma, y, cols, i);
    }
  }

  template <int64 ROWS>
  void dice_avx3(const float* x, const float* mean, const float* rvar,
                 const float* gamma, float* y, int64 cols, int64 row) {
    int64 j = 0;
    __m512 xs[ROWS];
    __m512 means[ROWS];
    __m512 rvars[ROWS];
    __m512 gammas[ROWS];
    __m512 exp_ts[ROWS];
    __m512 ys[ROWS];
    for (j; j < cols - 16; j += 16) {
      auto get_exp = [&](auto id) {
        xs[id] = _mm512_loadu_ps(INDEX_(x, id, j));
        means[id] = _mm512_loadu_ps(INDEX_1D(mean, j));
        rvars[id] = _mm512_loadu_ps(INDEX_1D(rvar, j));
        compute_exp(xs[id], means[id], rvars[id], exp_ts[id]);
      };
      compile_time_for<ROWS>::op(get_exp);

      auto get_px = [&](auto id) {
        gammas[id] = _mm512_loadu_ps(INDEX_1D(gamma, j));
        ys[id] = compute_px(exp_ts[id], gammas[id]);
      };
      compile_time_for<ROWS>::op(get_px);

      auto get_result = [&](auto id) {
        ys[id] = _mm512_mul_ps(ys[id], xs[id]);
        _mm512_storeu_ps(INDEX_(y, id, j), ys[id]);
      };
      compile_time_for<ROWS>::op(get_result);
    }

    if (j < cols) {
      __mmask16 mask = 0xFFFF >> (16 + j - cols);

      auto get_exp = [&](auto id) {
        xs[id] = _mm512_maskz_loadu_ps(mask, INDEX_(x, id, j));
        means[id] = _mm512_maskz_loadu_ps(mask, INDEX_1D(mean, j));
        rvars[id] = _mm512_maskz_loadu_ps(mask, INDEX_1D(rvar, j));
        compute_exp(xs[id], means[id], rvars[id], exp_ts[id]);
      };
      compile_time_for<ROWS>::op(get_exp);

      auto get_px = [&](auto id) {
        gammas[id] = _mm512_maskz_loadu_ps(mask, INDEX_1D(gamma, j));
        ys[id] = compute_px(exp_ts[id], gammas[id]);
      };
      compile_time_for<ROWS>::op(get_px);

      auto get_result = [&](auto id) {
        ys[id] = _mm512_mul_ps(ys[id], xs[id]);
        _mm512_mask_storeu_ps(INDEX_(y, id, j), mask, ys[id]);
      };
      compile_time_for<ROWS>::op(get_result);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Dice").Device(DEVICE_CPU).TypeConstraint<float>("T"), DiceOp<float>);
}  // namespace tensorflow

#endif  // AVX512F
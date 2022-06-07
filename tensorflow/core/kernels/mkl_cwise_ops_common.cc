/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef INTEL_MKL

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS
#include <iostream>
#include <vector>

#include "tensorflow/core/kernels/cwise_ops_common.h"

#include "tensorflow/core/util/mkl_util.h"

using dnnl::binary;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Primitive cache.
struct MklBinaryParams {
  memory::desc src0_md;
  memory::desc src1_md;
  dnnl::algorithm alg_kind;
  bool is_src0_blocked_format;
  bool is_src1_blocked_format;

  MklBinaryParams(memory::desc& src0_md, memory::desc& src1_md,
                  dnnl::algorithm alg_kind, bool is_src0_blocked_format,
                  bool is_src1_blocked_format)
      : src0_md(src0_md),
        src1_md(src1_md),
        alg_kind(alg_kind),
        is_src0_blocked_format(is_src0_blocked_format),
        is_src1_blocked_format(is_src1_blocked_format) {}
};

template <typename T>
class MklBinaryPrimitive : public MklPrimitive {
 public:
  explicit MklBinaryPrimitive(const MklBinaryParams& binary_params)
      : MklPrimitive(engine(ENGINE_CPU, 0)) {
    Setup(binary_params);
  }

  ~MklBinaryPrimitive() {}

  const std::shared_ptr<dnnl::binary::primitive_desc> GetBinaryPd() const {
    return context_.primitive_desc;
  }

  void Execute(const T* src0_data, const T* src1_data, T* dst_data,
               std::shared_ptr<stream> stream) {
    context_.src0_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src0_data)));
    context_.src1_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src1_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));

    execute_primitives(context_.primitives, stream, context_.primitives_args);

    // Clean up after operation execution.
    context_.src0_mem->set_data_handle(DummyData);
    context_.src1_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

 private:
  struct BinaryContext {
    std::shared_ptr<memory::desc> src0_md;
    std::shared_ptr<memory::desc> src1_md;
    std::shared_ptr<memory::desc> dst_md;

    std::shared_ptr<memory> src0_mem;
    std::shared_ptr<memory> src1_mem;
    std::shared_ptr<memory> dst_mem;

    std::shared_ptr<dnnl::binary::desc> desc;
    std::shared_ptr<dnnl::binary::primitive_desc> primitive_desc;
    std::shared_ptr<dnnl::primitive> primitive;

    std::vector<dnnl::primitive> primitives;
    std::vector<std::unordered_map<int, memory>> primitives_args;

    BinaryContext()
        : src0_md(nullptr),
          src1_md(nullptr),
          dst_md(nullptr),
          src0_mem(nullptr),
          src1_mem(nullptr),
          dst_mem(nullptr),
          desc(nullptr),
          primitive_desc(nullptr),
          primitive(nullptr) {}
  };

  void Setup(const MklBinaryParams& binary_params) {
    context_.src0_md.reset(new memory::desc(binary_params.src0_md.data));
    context_.src1_md.reset(new memory::desc(binary_params.src1_md.data));
    context_.dst_md.reset(new memory::desc(binary_params.src0_md.data));

    context_.src0_mem.reset(
        new memory(*context_.src0_md, cpu_engine_, DummyData));
    context_.src1_mem.reset(
        new memory(*context_.src1_md, cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(*context_.dst_md, cpu_engine_, DummyData));

    context_.desc.reset(
        new dnnl::binary::desc(binary_params.alg_kind, *context_.src0_md,
                                 *context_.src1_md, *context_.dst_md));

    context_.primitive_desc.reset(
        new dnnl::binary::primitive_desc(*context_.desc, cpu_engine_));

    context_.primitive.reset(new dnnl::binary(*context_.primitive_desc));

    std::unordered_map<int, memory> net_args;
    net_args.insert({DNNL_ARG_SRC_0, *context_.src0_mem});
    net_args.insert({DNNL_ARG_SRC_1, *context_.src1_mem});
    net_args.insert({DNNL_ARG_DST, *context_.dst_mem});

    context_.primitives_args.push_back(net_args);

    context_.primitives.push_back(*context_.primitive);
  }

  struct BinaryContext context_;
};

template <typename T>
class MklBinaryPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklBinaryPrimitive<T>* Get(const MklBinaryParams& binary_params,
                                    bool do_not_cache) {
    MklBinaryPrimitive<T>* primitive = nullptr;

    if (do_not_cache) {
      primitive = new MklBinaryPrimitive<T>(binary_params);
    } else {
      primitive = dynamic_cast<MklBinaryPrimitive<T>*>(
          MklBinaryPrimitiveFactory<T>::GetInstance().GetBinary(binary_params));
      if (!primitive) {
        primitive = new MklBinaryPrimitive<T>(binary_params);
        MklBinaryPrimitiveFactory<T>::GetInstance().SetBinary(binary_params,
                                                              primitive);
      }
    }

    return primitive;
  }

 private:
  MklBinaryPrimitiveFactory() {}
  ~MklBinaryPrimitiveFactory() {}

  static MklBinaryPrimitiveFactory& GetInstance() {
    static MklBinaryPrimitiveFactory instance_;
    return instance_;
  }

  static std::string CreateKey(const MklBinaryParams& binary_params) {
    std::string prefix = "binary_params_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(binary_params.src0_md.dims());
    key_creator.AddAsKey(binary_params.src1_md.dims());
    key_creator.AddAsKey(typeid(T).name());
    // Use blocked format info to avoid unnecessary reorder when inputs
    // have same shape but different formats.
    key_creator.AddAsKey(binary_params.is_src0_blocked_format);
    key_creator.AddAsKey(binary_params.is_src1_blocked_format);
    key_creator.AddAsKey(binary_params.alg_kind);
    return key_creator.GetKey();
  }

  MklPrimitive* GetBinary(const MklBinaryParams& binary_params) {
    std::string key = CreateKey(binary_params);
    return this->GetOp(key);
  }

  void SetBinary(const MklBinaryParams& binary_params, MklPrimitive* op) {
    std::string key = CreateKey(binary_params);
    this->SetOp(key, op);
  }
};

// Binary op.
template <typename Device, typename Functor, typename T>
class MklBinaryOp : public BinaryOp<Device, Functor> {
 public:
  explicit MklBinaryOp(OpKernelConstruction* context)
      : BinaryOp<Device, Functor>(context) {}

  void Compute(OpKernelContext* context) override {
    const int kNumInputs = 2;
    std::vector<Tensor> inputs(kNumInputs);
    std::vector<MklDnnShape> mkl_input_shapes(kNumInputs);
    std::vector<TensorShape> tf_input_shapes(kNumInputs);
    std::vector<T*> inputs_data(kNumInputs);

    for (int i = 0; i < kNumInputs; i++) {
      inputs[i] = context->input(i);
      inputs_data[i] = inputs[i].flat<T>().data();
      GetMklShape(context, i, &(mkl_input_shapes[i]));

      tf_input_shapes[i] = mkl_input_shapes[i].IsMklTensor()
                               ? mkl_input_shapes[i].GetTfShape()
                               : inputs[i].shape();
    }

    VLOG(1) << "MklBinaryOp: Inputs shapes " << tf_input_shapes[0].DebugString()
            << " _and_ " << tf_input_shapes[1].DebugString();

    if (!ShouldFallback(tf_input_shapes[0], tf_input_shapes[1])) {
      VLOG(1) << "MklBinaryOp: Run oneDNN primitive";

      std::vector<memory::dims> srcs_dims(kNumInputs);
      for (int i = 0; i < kNumInputs; i++) {
        auto& mkl_shape = mkl_input_shapes[i];
        if (mkl_shape.IsMklTensor()) {
          srcs_dims[i] = mkl_shape.GetSizesAsMklDnnDims();
        } else {
          srcs_dims[i] = TFShapeToMklDnnDims(tf_input_shapes[i]);
        }
      }

      // oneDNN only supports inputs[1] bcast to inputs[0]. So if inputs[1]
      // has more elements than inputs[0], swap the 2 inputs.
      // Use an index to indicate the swapped result.
      const int kFirst = (tf_input_shapes[1].num_elements() >
                          tf_input_shapes[0].num_elements()) ||
                         (tf_input_shapes[1].num_elements() ==
                              tf_input_shapes[0].num_elements() &&
                          srcs_dims[1].size() > srcs_dims[0].size());
      const int kSecond = 1 - kFirst;

      // oneDNN only supports inputs with same rank size, so expand dimension
      // if they are not consistent.
      // E.g. 8x4 * 4 --> 8x4 * 1x4.
      if (srcs_dims[0].size() != srcs_dims[1].size()) {
        const int kSmall = srcs_dims[0].size() > srcs_dims[1].size();
        ExpandDim(srcs_dims[kSmall], srcs_dims[1 - kSmall].size());
      }

      std::vector<memory::desc> srcs_md(kNumInputs);
      for (int i = 0; i < kNumInputs; i++) {
        auto& mkl_shape = mkl_input_shapes[i];
        if (mkl_shape.IsMklTensor()) {
          srcs_md[i] = mkl_shape.GetMklLayout();
        } else {
          auto src_strides = CalculateTFStrides(srcs_dims[i]);
          srcs_md[i] =
              MklDnnData<T>::CreateBlockedMemDesc(srcs_dims[i], src_strides);
        }
      }

      // Get algorithm kind from Functor.
      dnnl::algorithm alg_kind = dnnl::algorithm::undef;

      if (std::is_same<Functor, functor::add<T>>::value) {
        alg_kind = ALGORITHM::binary_add;
      } else if (std::is_same<Functor, functor::mul<T>>::value) {
        alg_kind = ALGORITHM::binary_mul;
      } else if (std::is_same<Functor, functor::maximum<T>>::value) {
        alg_kind = ALGORITHM::binary_max;
      } else {
        OP_REQUIRES_OK(context,
                       errors::Aborted("Unsupported oneDNN binary algorithm"));
      }

      MklBinaryParams params(srcs_md[kFirst], srcs_md[kSecond], alg_kind,
                             mkl_input_shapes[kFirst].IsMklTensor(),
                             mkl_input_shapes[kSecond].IsMklTensor());

      MklBinaryPrimitive<T>* primitive =
          MklBinaryPrimitiveFactory<T>::Get(params, 0);
      auto binary_pd = primitive->GetBinaryPd();

      // Fallback to Eigen if it's in oneDNN slow `ref` path.
      if (IsPrimitiveRefPath(*binary_pd)) {
        VLOG(1) << "MklBinaryOp: Hit oneDNN `ref` path, fallback to Eigen ";

        FallbackToEigen(context);
        auto out = context->mutable_output(0);
        VLOG(1) << "MklBinaryOp: Ouput shapes " << out->shape().DebugString();

        return;
      }

      std::shared_ptr<stream> stream;
      MklDnnThreadPool eigen_tp(context);
      stream.reset(CreateStream(&eigen_tp, primitive->GetEngine()));

      // TODO(intel): Do inplace optimization if meet performance issue.
      MklDnnShape dnn_shape_dst;
      TensorShape tf_shape_dst;
      AllocateOutputTensor(*binary_pd, tf_input_shapes[kFirst],
                           mkl_input_shapes[kFirst], tf_shape_dst,
                           dnn_shape_dst);
      Tensor* dst_tensor = nullptr;
      AllocateOutputSetMklShape(context, 0, &dst_tensor, tf_shape_dst,
                                dnn_shape_dst);
      DCHECK(dst_tensor != nullptr) << "Output tensor pointer is NULL";

      // Reorder inputs if necessary.
      std::vector<MklDnnData<T>> mkl_srcs(kNumInputs,
                                          MklDnnData<T>(&cpu_engine_));
      for (int i = 0; i < kNumInputs; i++) {
        // Map swapped result to current index.
        const int cur_index = std::abs(kFirst - i);

        if (binary_pd->src_desc(i) != srcs_md[cur_index]) {
          mkl_srcs[i].SetUsrMem(srcs_md[cur_index], inputs_data[cur_index]);
          mkl_srcs[i].CheckReorderToOpMem(binary_pd->src_desc(i), cpu_engine_,
                                          context);
          inputs_data[cur_index] =
              reinterpret_cast<T*>(mkl_srcs[i].GetOpMem().get_data_handle());
        }
      }

      primitive->Execute(inputs_data[kFirst], inputs_data[kSecond],
                         dst_tensor->flat<T>().data(), stream);
    } else {
      VLOG(1) << "MklBinaryOp: Fall back to Eigen";

      FallbackToEigen(context);
    }

    auto out = context->mutable_output(0);
    VLOG(1) << "MklBinaryOp: Ouput shapes " << out->shape().DebugString();
  }

  engine cpu_engine_ = engine(ENGINE_CPU, 0);

 private:
  // Expand dimension size to `max_dim`, the new dimensions will be added to
  // left: 4 -- > 1x4.
  // It follows Numpy broadcast rule:
  // http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
  void ExpandDim(memory::dims& dims, int max_dim) {
    if (dims.size() >= max_dim) return;

    int start_offset = max_dim - dims.size();
    std::vector<int> expanded(max_dim, 1);
    for (int i = 0; i < dims.size(); i++) {
      expanded[start_offset + i] = dims[i];
    }

    dims.resize(max_dim);
    for (int i = 0; i < max_dim; i++) {
      dims[i] = expanded[i];
    }
  }

  void FallbackToEigen(OpKernelContext* context) {
    BinaryOp<Device, Functor>::Compute(context);

    // OneDNN metadata inputs won't be changed in Eigen path, direct forward them.
    ForwardMklMetaDataInToOut(context, 0, 0);
  }

  bool ShouldFallback(const TensorShape& shape0, const TensorShape& shape1) {
    // oneDNN doesn't support Sub and SuqaredDiff yet.
    if (std::is_same<Functor, functor::sub<T>>::value ||
        std::is_same<Functor, functor::squared_difference<T>>::value) {
      return true;
    }

    if (UnsupportShape(shape0, shape1)) return true;

    return false;
  }

  bool UnsupportShape(const TensorShape& shape0, const TensorShape& shape1) {
    // Bi-bcast like 8x1 * 1x4 isn't supported in oneDNN. Compare output
    // shape(8x4) with input shapes, and fall back to Eigen if output has more
    // elements than all inputs.
    int64 dst_elements = 1;
    TensorShape l = shape0.dims() > shape1.dims() ? shape0 : shape1;
    TensorShape s = shape0.dims() > shape1.dims() ? shape1 : shape0;
    int gap = l.dims() - s.dims();
    for (int i = 0; i < gap; ++i) dst_elements *= l.dim_size(i);
    for (int i = 0; i < s.dims(); ++i)
      dst_elements *= std::max(s.dim_size(i), l.dim_size(i + gap));

    if (dst_elements > shape0.num_elements() &&
        dst_elements > shape1.num_elements())
      return true;

    // Eigen will fill specific shape to output when **the** input shape is 0,
    // oneDNN does not handle this case.
    if (shape0.dims() == 0 || shape1.dims() == 0 ||
        shape0.num_elements() == 0 || shape1.num_elements() == 0)
      return true;

    // Currently oneDnn can only support up to 5 dimensions.
    if (shape0.dims() > 5 || shape1.dims() > 5) return true;

    return false;
  }

  void AllocateOutputTensor(const binary::primitive_desc& binary_pd,
                            const TensorShape& tf_shape_src,
                            const MklDnnShape& dnn_shape_src,
                            TensorShape& tf_shape_dst,
                            MklDnnShape& dnn_shape_dst) {
    if (dnn_shape_src.IsMklTensor()) {
      dnn_shape_dst.SetMklTensor(true);
      auto dst_pd = binary_pd.dst_desc();
      dnn_shape_dst.SetMklLayout(&dst_pd);
      dnn_shape_dst.SetElemType(MklDnnType<T>());
      dnn_shape_dst.SetTfLayout(dnn_shape_src.GetDimension(),
                                dnn_shape_src.GetSizesAsMklDnnDims(),
                                dnn_shape_src.GetTfDataFormat());
      // Reshape the output to 1-D tensor if input is blocked format.
      tf_shape_dst.AddDim(dst_pd.get_size() / sizeof(T));
    } else {
      dnn_shape_dst.SetMklTensor(false);
      tf_shape_dst = tf_shape_src;
    }
  }
};

//---------- Registration macros for various element-wise ops -----------
// We will need to redefine "REGISTER" to include the mkl_op_registry flag
#pragma push_macro("REGISTER")
#undef REGISTER
#define REGISTER(OP, D, N, F, T)                               \
  REGISTER_KERNEL_BUILDER(                                     \
      Name(N)                                                  \
          .Device(DEVICE_##D)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      OP<D##Device, F<T>, T>);

REGISTER2(MklBinaryOp, CPU, "_MklAdd", functor::add, float, bfloat16);
REGISTER2(MklBinaryOp, CPU, "_MklAddV2", functor::add, float, bfloat16);
REGISTER2(MklBinaryOp, CPU, "_MklSub", functor::sub, float, bfloat16);
REGISTER2(MklBinaryOp, CPU, "_MklMul", functor::mul, float, bfloat16);
REGISTER2(MklBinaryOp, CPU, "_MklMaximum", functor::maximum, float, bfloat16);
REGISTER2(MklBinaryOp, CPU, "_MklSquaredDifference",
          functor::squared_difference, float, bfloat16);

#undef REGISTER
#pragma pop_macro("REGISTER")
//-----------------------------------------------------------------------

}  // end namespace tensorflow

#endif  // INTEL_MKL

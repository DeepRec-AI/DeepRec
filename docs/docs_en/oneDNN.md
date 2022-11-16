# oneDNN

## Introduction

[oneDNN](https://github.com/oneapi-src/oneDNN) is the open source cross-platform performance acceleration library for deep learning from Intel, The [documentation](https://oneapi-src.github.io/oneDNN/) guides you to find out which primitives are supported. OneDNN has been integrated into DeepRec, which can be enabled by adding the compiling option in the compile command. `--config=mkl_threadpool` is used to enable oneDNN accelerated arithmetic computation. Adding the compiling option `--config=opt` will enable the optimization of `--copt=-march=native`, which can further accelerate arithmetic performance on the CPU which supports AVX512, for example, Skylake, Caslake and Icelake.



Tips: MKL was first renamed as DNNL and then renamed as oneDNN. Tensorflow initially used MKL to accelerate the computation of the operators, and in subsequent versions of iteration, oneDNN gradually take the place of MKL, but the macro definitions were still retained. 



Macro definition of oneDNN in DeepRec:

| Macro Definition                 |  Values（Bold for Default）            | Explanation                                                  |
| :------------------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| TF_MKL_PRIMITIVE_ONLY_FOR_RECO   | **1/true**, 0/false                           | 1: Only replace the [operators](https://github.com/alibaba/DeepRec/blob/main/tensorflow/core/graph/mkl_layout_pass.cc#L824-L840) which supported by oneDNN in recommendation models; 0: Replace all of the operators to that supported by oneDNN. |
| TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE | **1/true**, 0/false                           | 1: Reduce the use of main memory by releasing the primitives; 0: Don't release primitives. |
| TF_DISABLE_MKL                   | **0**, 1                                      | 0: Enable MKL; 1: Disable MKL                                |
| TF_MKL_NUM_INTRAOP               | Integer, such as 14 ,**Not set by default**   | Integer：set the number of intra threads used by oneDNN；Not set：number of TF intra threads used most. |
| ONEDNN_VERBOSE                   | **0**/1/2                                     | Print the [level](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html) of log output by oneDNN primitive. |
| DNNL_MAX_CPU_ISA                 | **ALL**, AVX512_CORE_AMX, AVX512_CORE_BF16, … | The[ highest ISA](https://oneapi-src.github.io/oneDNN/v2.4/dev_guide_cpu_dispatcher_control.html#run-time-controls) used by oneDNN (for versions less than 2.5.0) |
| ONEDNN_MAX_CPU_ISA               | **ALL**, AVX512_CORE_AMX, AVX512_CORE_BF16, … | The [highest ISA](https://oneapi-src.github.io/oneDNN/v2.4/dev_guide_cpu_dispatcher_control.html#run-time-controlsused) by oneDNN (for versions more than or equal to 2.5.0) |

Primitives supported by oneDNN:

| Primitive                              | Available Types               | Available Backward Operations     |
| -------------------------------------- | --------------------------- | --------------------------------- |
| Matrix Multiplication                  | f32, bf16, f16, u8, s8      | Scale, Zero, Eltwise, Sum, Binary |
| Inner Product                          | f32, bf16, f16, u8, s8      | Scale, Eltwise, Sum, Binary       |
| Layer Normalization                    | f32, bf16, f16              | /                                 |
| Batch Normalization                    | f32, bf16, f16, s8          | Eltwise                           |
| Local Response Normalization (LRN)     | f32, bf16, f16              | /                                 |
| Binary (+, =, *, /, >, <, min, max...) | f32, bf16, f16, u8, s8      | Scale, Eltwise, Sum, Binary       |
| Eltwise (relu, gelu, tanh, linear...)  | f32, s32, bf16, f16, u8, s8 | Binary                            |
| PReLU                                  | f32, s32, bf16, s8, u8      | /                                 |
| Sum                                    | f32, s32, bf16, f16, u8, s8 | /                                 |
| Reduction                              | f32, bf16, u8, s8           | Eltwise, Sum, Binary              |
| Softmax                                | f32, bf16, f16              | /                                 |
| LogSoftmax                             | f32, bf16                   | /                                 |
| Reorder                                | f32, s32, bf16, f16, u8, s8 | Scale, Sum                        |
| Concat                                 | f32, s32, bf16, f16, u8, s8 | /                                 |
| Convolution                            | f32, bf16, f16, u8, s8      | Scale, Zero, Eltwise, Sum, Binary |
| Pooling                                | f32, s32, bf16, f16, u8, s8 | Binary                            |
| RNN (LSTM, GRU, Vanilla RNN...)        | f32, bf16, f16, u8, s8      | /                                 |
| Resampling                             | f32, s32, bf16, f16, s8, u8 | Eltwise, Sum, Binary              |
| Shuffle                                | f32, s32, bf16, s8, u8      | /                                 |


<h1 align="center">
    DeepRec
</h1>

# Introduction

DeepRec is a recommendation engine based on TensorFlow 1.15, Intel-TensorFlow and NVIDIA-TensorFlow.

# Background

Sparse model is a type of deep learning model that accounts for a relatively high proportion of discrete feature calculation logic in the model structure. Discrete features are usually expressed as non-numeric features that cannot be directly processed by algorithms such as id, tag, text, and phrases. They are widely used in high-value businesses such as search, advertising, and recommendation.

DeepRec has been deeply cultivated since 2016, which supports core businesses such as Taobao Search, recommendation and advertising. It precipitates a list of features on basic frameworks and has excellent performance in sparse models training. Facing a wide variety of external needs and the environment of deep learning framework embracing open source, DeepeRec open source is conducive to establishing standardized interfaces, cultivating user habits, greatly reducing the cost of external customers working on cloud and establishing the brand value.

# Getting started

```{toctree}
:maxdepth: 2
:caption: Build & Install
DeepRec-Compile-And-Install
Estimator-Compile-And-Install
TFServing-Compile-And-Install
```

# Features

```{toctree}
:maxdepth: 2
:caption: Distributed Training

GRPC++
StarServer
SOK
```

```{toctree}
:maxdepth: 2
:caption: Graph Optimization
Fused-Embedding
Sample-awared-Graph-Compression
```

```{toctree}
:maxdepth: 2
:caption: Runtime Optimization

CPU-Memory-Optimization
GPU-Memory-Optimization
GPU-Virtual-Memory
Executor-Optimization
GPU-MultiStream
```

```{toctree}
:maxdepth: 2
:caption: Model Save & Restore

```

```{toctree}
:maxdepth: 2
:caption: Optimizer

```

```{toctree}
:maxdepth: 2
:caption: Operator & Hardware Acceleration

oneDNN
Operator-Optimization
NVIDIA-TF32
```

```{toctree}
:maxdepth: 2
:caption: Model Quantification

BFloat16
```

```{toctree}
:maxdepth: 2
:caption: Dataset and Processing

```

```{toctree}
:maxdepth: 2
:caption: Compiler

```

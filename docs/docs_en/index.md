<h1 align="center">
    DeepRec
</h1>

# Introduction

DeepRec is a recommendation engine based on TensorFlow 1.15, Intel-TensorFlow and NVIDIA-TensorFlow.

# Background

Sparse model is a type of deep learning model that accounts for a relatively high proportion of discrete feature calculation logic in the model structure. Discrete features are usually expressed as non-numeric features that cannot be directly processed by algorithms such as id, tag, text, and phrases. They are widely used in high-value businesses such as search, advertising, and recommendation.

DeepRec has been deeply cultivated since 2016, which supports core businesses such as Taobao Search, recommendation and advertising. It precipitates a list of features on basic frameworks and has excellent performance in sparse models training. Facing a wide variety of external needs and the environment of deep learning framework embracing open source, DeepeRec open source is conducive to establishing standardized interfaces, cultivating user habits, greatly reducing the cost of external customers working on cloud and establishing the brand value.

# Getting started

# Features
```{toctree}
:maxdepth: 2
:caption: Operator & Hardware Acceleration

oneDNN
Operator-Optimization
```

```{toctree}
:maxdepth: 2
:caption: Model Quantification

BFloat16
```

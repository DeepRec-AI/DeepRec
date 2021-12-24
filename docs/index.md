<h1 align="center">
    DeepRec
</h1>

# Introduction
DeepRec is a recommendation engine based on [TensorFlow 1.15](https://www.tensorflow.org/), [Intel-TensorFlow](https://github.com/Intel-tensorflow/tensorflow) and [NVIDIA-TensorFlow](https://github.com/NVIDIA/tensorflow).


Sparse model is a type of deep learning model that accounts for a relatively high proportion of discrete feature calculation logic in the model structure. Discrete features are usually expressed as non-numeric features that cannot be directly processed by algorithms such as id, tag, text, and phrases. They are widely used in high-value businesses such as search, advertising, and recommendation.


DeepRec has been deeply cultivated since 2016, which supports core businesses such as Taobao Search, recommendation and advertising. It precipitates a large number of optimized operators on basic frameworks and has excellent performance in sparse models training.Facing a wide variety of external needs and the environment of deep learning framework embracing open source, deeperec open source is conducive to establishing standardized interfaces, cultivating user habits, greatly reducing the cost of external customers working on cloud and establishing the brand value.


DeepRec has super large-scale distributed training capability, supporting model training of trillion samples and 100 billion Embedding Processing. For sparse model scenarios, in-depth performance optimization has ben conducted across CPU and GPU platform. It contains 3 kinds of key features to improve usability and performance for super-scale scenarios. 
# Contents

```{toctree}
:maxdepth: 2

Embedding-Variable
Feature-Eviction
Dynamic-dimension-Embedding-Variable
Adaptive-Embedding
Multi-Hash-Variable
GRPC++
StarServer
Auto-Micro-Batch
Fused-Embedding
Smart-Stage
TensorPoolAllocator
WorkQueue
Incremental-Checkpoint
AdamAsync-Optimizer
AdagradDecay-Optimizer
NVIDIA-TF32
oneDNN
```

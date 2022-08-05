# Release r1.15.5-deeprec2201

This is the first release of DeepRec. DeepRec has super large-scale distributed training capability, supporting model training of trillion samples and 100 billion Embedding Processing. For sparse model scenarios, in-depth performance optimization has been conducted across CPU and GPU platform.

## **Major Features and Improvements**

### **Embedding**
- Embedding Variable (including feature eviction and feature filter)
- Dynamic Dimension Embedding Variable
- Adaptive Embedding
- Multi-Hash Variable

### **Distributed Training**
- GRPC++
- StarServer
- Synchronous Training - SOK

### **Graph Optimization**
- Auto Micro Batch
- Auto Graph Fusion
- Embedding Fusion
- Smart Stage

### **Runtime Optimization**
- CPU Memory Optimization
- GPU Memory Optimization
- GPU Virtual Memory

### **Model Export**
- Incremental Checkpoint

### **Optimizer**
- AdamAsync Optimizer
- AdagradDecay Optimizer
- AdamW Optimizer

### **Op & Hardware Acceleration**
- Operators Optimization: Unique, Gather, DynamicStitch, BiasAdd, Select, Transpose, SparseSegmentReduction, where, DynamicPartition, SparseConcat tens of ops' CPU/GPU optimization.
- support oneDNN & BFloat16(BF16) & Advanced Matrix Extension(AMX)
- Support TensorFloat-32(TF32)

### **IO & Dataset**
- WorkQueue
- KafkaDataset
- KafkaGroupIODataset 

More details of features: [DeepRec Document]([url](https://deeprec.rtfd.io))

## **Release Images**

### **CPU Image**
`registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-training:deeprec2201-cpu-py36-ubuntu18.04`

### **GPU Image**
`registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-training:deeprec2201-gpu-py36-cu110-ubuntu18.04`

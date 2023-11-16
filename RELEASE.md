# Release r1.15.5-deeprec2310
## **Major Features and Improvements**

### **Embedding**

- Refactor the data structure of EmbeddingVariable.
- Add interface of EmbeddingVar for Elastic Training.
- Add GetSnapshot and Create API for EmbeddingVariable.
- Remove the dependency on private header file in EmbeddingVariable.

### **Runtime Optimization**

- Canonicalize SaveV2 Op device spec in distributed training.
- Update log level in direct_session.

### **Distributed**

- Add elastic-grpc server.

### **BugFix**

- Fix missing return value of RestoreSSD of DramSSDHashStorage.
- Fix incorrect frequency in shared-embedding.
- Fix set initialized flag too early in restore subgraph.
- Fix wgrad bug in Sparse Operation Kit.
- Fix hang bug for async embedding lookup.
- Fix ps address list sort by index.
- Fix SharedEmbeddingColumn with PartitionedEmbedingVariable shape validation error.

More details of features: [https://deeprec.readthedocs.io/zh/latest/](url)

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2310-cpu-py38-ubuntu20.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2310-gpu-py38-cu116-ubuntu20.04`

# Release r1.15.5-deeprec2306

## **Major Features and Improvements**

### **Embedding**

- Support StaticGPUHashMap to optimize EmbeddingVariable in inference.
- Update logic of GroupEmbedding in feature_column API.
- Refine APIs for foward-backward optimization.
- Move insertions of new features into the backward process when lti-tier storage. 
- Move insertion of new features into the backward ops.
- Modify calculation logic of embedding lookup sparse combiner.
- Add memory and performance tests of EmbeddingVariable.

### **Graph & Grappler Optimization**

- Support IteratorGetNext for SmartStage as a starting node for searching.
- Reimplement PrefetchRunner in C++.

### **Runtime Optimization**

- Dispatch expensive ops via multiple threads in theadpool.
- Enable multi-stream in session_group by default.
- Support for loading saved_model with device information when use p and multi_stream.
- Make ARENA_ARRAY_SIZE to be configurable.
- Optimize EV allocator performance.
- Integrate HybridBackend in collective training mode.

### **Ops & Hardware Acceleration**

- Disable MatMul fused with LeakyRule when MKL is disabled.

### **Serving**

- Clear virtual_device configurations before load new checkpoint.

### **Environment & Build**

- Update docker images in user documents.
- Update DEFAULT_CUDA_VERSION and DEFAULT_CUDNN_VERSION in configure.py.
- Move thirdparties from WORKSPACE to workspace.bzl.
- Update urls corresponding to colm, ragel, aliyun-oss-sdk and uuid.
- Update default TF_CUDA_COMPUTE_CAPABILITIES to 7.0,7.5,8.0,8.6.
- Update SparseOperationKit to v23.5.01 and docker file.

### **BugFix**

- Fix issue of missing params while constructing the ngScope.
- Fix memory leak to avoid OOM.
- Fix shape validation in API shared_embedding_columns.
- Fix the device placement bug of stage_subgraph_on_cpu in distributed.
- Fix hung issue when using both SOK and SmartStaged simultaneously.
- Fix bug: init global_step before saving variables
- Fix bug: reserve input nodes, clear saver devices on demand.
- Fix memory leak when a graph node is invalid.

### **ModelZoo**

- Add examples and docs to demonstrate Collective Training.
- Update documents and config files for modelzoo benchmark.
- Update modelzoo README.

### **Tool & Documents**

- Update cases of configure TF_CUDA_COMPUTE_CAPABILITIES for H100.
- Update COMMITTERS.md.
- Update device placement documents.
- Update document for SmartStage.
- Update session_group documents.
- Update the download link of the library that Processor depends on.
- Update sok to 1.20.

More details of features: [https://deeprec.readthedocs.io/zh/latest/](url)

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2306-cpu-py38-ubuntu20.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2306-gpu-py38-cu116-ubuntu20.04`

# Release r1.15.5-deeprec2304

## **Major Features and Improvements**

### **Embedding**

- Suport tf.int32 dtype using feature_column API `tf.feature_column.categorical_column_with_embedding`.
- Make the rules of export frequencies and versions the same as the rule of export keys.
- Optimize cuda kernel implementation in GroupEmbedding.
- Support to read embedding files with mmap and madvise, and direct IO.
- Add double check in find_wait_free of lockless dense hashmap.
- Change Embedding init value of version in EV from 0 to -1.
- Interface 'GetSnapshot()' backward compatibility.
- Implement CPU GroupEmbedding lookup sparse Op.
- Make GroupEmbedding compatible with sequence feature_column interface.
- Fix sp_weights indices calculation error in GroupEmbedding.
- Add group_strategy to control parallelism of group_embedding.

### **Graph & Grappler Optimization**

- Support SparseTensor as placeholder in Sample-awared Graph Compression.
- Add Dice fusion grappler and ops.
- Enable MKL Matmul + Bias + LeakyRelu fusion.

### **Runtime Optimization**

- Avoid unnecessary polling in EventMgr.
- Reduce lock cost and memory usage in EventMgr when use multi-stream.

### **Ops & Hardware Acceleration**

- Register GPU implementation of int64 type for Prod.
- Register GPU implementation of string type for Shape, ShapeN and ExpandDims.
- Optimize list of GPU SegmentReductionOps.
- Optimize zeros_like_impl by reducing calls to convert_to_tensor.
- Implement GPU version of SparseSlice Op.
- Delay Reshape when rank > 2 in keras.layers.Dense so that post op can be fused with MatMul.
- Implement setting max_num_threads hint to oneDNN at compile time.
- Implement TensorPackTransH2DOp to improve SmartStage performance on GPU.

### **IO**

- Add tensor shape meta-data support for ParquetDataset.
- Add arrow BINARY type support for ParquetDataset.

### **Serving**

- Add Dice fusion to inference mode.
- Enable INFERENCE_MODE in processor.
- Support TensorRT 8.x in Inference.
- Add configure filed to control enable TensorRT or not.
- Add flag for device_placement_optimization.
- Avoid to clustering feature column related nodes when enable TensorRT.
- Optimize inference latency when load increment checkpoint.
- Optimize performance via only place TensorRT ops to gpu device.

### **Environment & Build**

- Support CUDA 12.
- Update DEFAULT_CUDA_VERSION and DEFAULT_CUDNN_VERSION in configure.py.
- Move thirdparties from WORKSPACE to workspace.bzl.
- Update urls corresponding to colm, ragel, aliyun-oss-sdk and uuid.

### **BugFix**

- Fix constant op placing bug for device placement optimization.
- Fix Nan issue occurred in group_embedding API.
- Fix SOK not compatible with variable issue.
- Fix memory leak when update full model in serving.
- Fix 'cols_to_output_tensors' not setted issue in GroupEmbedding.
- Fix core dump issue about saving GPU EmbeddingVariable.
- Fix cuda resource issue in KvResourceImportV3 kernel.
- Fix loading signature_def with coo_sparse bug and add UT.
- Fix the bug that the training ends early when the workqueue is enabled.
- Fix the control edge connection issue in device placement optimization.

### **ModelZoo**

- Modify GroupEmbedding related function usage.
- Update masknet example with layernorm.

### **Tool & Documents**

- Add tools for remove filtered features in checkpoint.
- Add Arm Compute Library (ACL) user documents.
- Update Embedding Variable document to fix initializer config example.
- Update GroupEmbedding document.
- Update processor documents.
- Add user documents for intel AMX.
- Add TensorRT usage documents.
- Update documents for ParquetDataset.

More details of features: [https://deeprec.readthedocs.io/zh/latest/](url)

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2304-cpu-py38-ubuntu20.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2304-gpu-py38-cu116-ubuntu20.04`

# Release r1.15.5-deeprec2302

## **Major Features and Improvements**

### **Embedding**

- Support same saver graph for EmbeddingVariable on GPU/CPU devices.
- Support save and restore parameters in HBM storage of EmbeddingVariable.
- Add GPU apply ops of Adam, AdamAsync, AdamW for multi-tier storage of EmbeddingVariable.
- Place output of KvResourceIsInitializedOp on CPU.
- Support GroupEmbedding to pack multiple feature columns lookup/apply.
- Optimize HBM-DRAM storage of EmbeddingVariable with intra parallelism and fine-grained synchronization.
- Support not saving filtered features when saving checkpoint.
- Support localized mode fusion in GroupEmbedding.
- Support to avoid preloaded IDs being eliminated in multi-tier embedding's cache.
- Support COMPACT layout to reduce memory cost in EmbeddingVariable.
- Support to ignore version when restore Embedding Variable with TF_EV_RESET_VERSION.
- Support restore custom dimension of Embedding Variable.
- Support merge and delete checkpoint files of SSDHash storage.

### **Graph & Grappler Optimization**

- Optimize SmartStage by prefetching LookupID op.
- Decouple SmartStage and forward backward joint optimization.
- Support Sample-awared Graph Compression.
- Support CUDA multi-stream for Stage.
- Improve Device Placement Optimization performance.
- Add TensorBufferPutGpuOp to improve SmartStage performance on GPU device.

### **Runtime Optimization**

- Enable EVAllocator by default.
- Optimize executor to eliminate sort latency and reduce memory.

### **Ops & Hardware Acceleration**

- Add list of GPU Ops for forward backward joint optimization.
- Optimize FusedBatchNormGrad on CPU device.
- Support NCHW format input for FusedBatchNormOp.
- Use new asynchronous evaluation in Eigen to FusedBatchNorm.
- Add exponential_avg_factor attribute to FusedBatchNorm* kernels.
- Change AliUniqueGPU kernel implementation to AsyncOpKernel.
- Support computing exponential running mean and variance in fused_batch_norm.
- Upgrade oneDNN to 2.7 and ACL to 22.08.
- Use global cache for MKL primitives for ARM.
- Disable optimizing batch norm as sequence of post ops on AArch64.
- Restore re-mapper and fix BatchMatmul and FactoryKeyCreator under AArch64 + ACL.

### **Distributed**

- Speedup SOK by GroupEmbedding which fuse multiple feature column together.

### **Serving**

- Support to setup gpu config in SessionGroup.
- Support to use multiple GPUs in SessionGroup.
- Support processor to set multi-stream option.
- Add flag to disable per_session_host_allocator.
- Run init_op on all sessions in session_group.
- Skip invalid request and return error msg to client.
- Use graph signature as the key to get runtime executor.

### **Environment & Build**

- Optimize compile time for kv_variable_ops module.
- Add dataset headers for custom op compilation.
- Add docker images for ARM based on ubuntu22.04.
- Upgrade BAZEL version to 3.7.2.

### **BugFix**

- Do not cudaSetDevice to invisible GPU in CreateDevices.
- Fix concurrency issue caused by not reference to same lock in multi-tier storage.
- Fix parse input request bug.
-  Fix the bug when saving empty GPU EmbeddingVariable. 
- Fix the concurrency issue between feature eviction and embedding lookup in asynchronous training.

### **ModelZoo**

- Support Parquet Dataset in list of models.
- Add GPU benchmark in Modelzoo.
- Unify the usage of price column in Taobao dataset.
- Add DeepFM model with int64 categorical id input.
- Update dataset url in Modelzoo.

### **Tool & Documents**

- Add checkpoint meta transformer tool. 
- Add list of user documents in English.

More details of features: [https://deeprec.readthedocs.io/zh/latest/](url)

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2302-cpu-py38-ubuntu20.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2302-gpu-py38-cu116-ubuntu20.04`

# Release r1.15.5-deeprec2212u1

## **Major Features and Improvements**

### **BugFix**

- Add flag to disable per_session_host_allocator.
- Fix bug of saving EmbeddingVariable with int32 type.
- Revert "Support fused batchnorm with any ndims and axis".

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2212u1-cpu-py38-ubuntu20.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2212u1-gpu-py38-cu116-ubuntu20.04`

# Release r1.15.5-deeprec2212

## **Major Features and Improvements**

### **Embedding**

- Refactor GPU Embedding Variable storage layer.
- Remove TENSORFLOW_USE_GPU_EV macro from embedding storage layer.
- Refactor KvResourceGather GPU Op. 
- Add embedding memory pool for HBM storage of EmbeddingVariable.
- Refine the code HBM storage of EmbeddingVariable.
- Reuse the embedding files on SSD generated by EmbeddingVariable when save and restore checkpoint.
- Integrate single HBM EV into multi_tier EmbeddingVariable.

### **Graph & Grappler Optimization**

- Filter out the 'stream_id' attribute in arithmetic optimizer.
- Add SimplifyEmbeddingLookupStage optimizer.
- Add ForwardBackwardJointOptimizationPass to eliminate duplicate hash in Gather and Apply ops for Embedding Variable.

### **Runtime Optimization**

- Add allocators for each stream_executor in multi-context mode.
- Set multi-gpu devices in session_group mode. 
- Add blacklist and whitelist to JitCugraph.
- Optimize CPU EVAllocator to speedup EmbeddingVariable performance.
- Support independent GPU host allocator for each session.
- Add GPU EVAllocator to speedup EmbeddingVariable on GPU.

### **Ops & Hardware Acceleration**

- Add GPU implementation for Unique.
- Support indices type with DT_INT64 in sparse segment ops.
- Add list of gradient implementation for the following ops including SplitV, ConcatV2, BroadcastTo, Tile, GatherV2, Cumsum, Cast.
- Add C++ gradient op for Select.
- Add gradient implementation for SelectV2.
- Add C++ gradient op for Atan2.
- Add C++ gradients for UnsortedSegmentMin/Max/Sum.
- Refactor KvSparseApplyAdagrad GPU Op.
- Merge NV-TF r1.15.5+22.12.

### **Distributed**

- Update seastar to control SDT by macro HAVE_SDT. 
- Update WORKER_DEFAULT_CORE_NUM(8) and PS_EFAULT_CORE_NUM(2) default values.

### **Serving**

- Support multi-model deployment in SessionGroup.
- Support user setup cpu-sets for each session_group. 
- Support processor to load multi-models.
- Support GPU compilation in processor.
- Optimize independent GPU host allocator for each session.

### **Environment & Build**

- Update systemtap to a valid source address. 
- Support DeepRec's ABI compatible with TensorFlow 1.15 by configure TF_API_COMPATIBLE_1150.
- Upgrade base docker images based on ubuntu20.04 and python3.8.10.
- Update pcre-8.44 urls.
- Remove systemtap from third party and related dependency.
- Enable gcc optimization option -O3 by default.

### **BugFix**

- Fix function definition issue in processor.
- Fix the hang when insert item into lockless hash map.
- Fix EmbeddingVariable hang/coredump in GPU mode.
- Fix memory leak in CUDA multi-stream when merge compute and copy stream.
- Fix wrong session devices order.
- Fix hwloc build error on alinux3.
- Fix double clear resource_mgr bug when use SessionGroup.
- Fix wrong Shrink causes unit tests to fail randomly.
- Fix the conflict when the EmbeddingVariable and embedding fusion is enabled simultaneously.
- Fix EmbeddingVarGPU coredump in destructor.

More details of features: [https://deeprec.readthedocs.io/zh/latest/](url)

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2212-cpu-py38-ubuntu20.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2212-gpu-py38-cu116-ubuntu20.04`

# Release r1.15.5-deeprec2210

## **Major Features and Improvements**

### **Embedding**

- Support HBM-DRAM-SSD storage in EmbeddingVariable multi-tier storage.
- Support multi-tier EmbeddingVariable initialized based on frequency when restore model.
- Support to lookup location of ids of EmbeddingVariable.
- Support kv_initialized_op for GPU Embedding Variable.
- Support restore compatibility of EmbeddingVariable using init_from_proto.
- Improve performance of apply/gather ops for EmbeddingVariable.
- Add Eviction Manager in EmbeddingVariable Multi-tier storage.
- Add unified thread pool for cache of Multi-tier storage in EmbeddingVariable.
- Save frequencies and versions of features in SSDHash and LevelDB storage of EmbeddingVariable.
- Avoid invalid eviction use HBM-DRAM storage of EmbeddingVariable.
- Preventing from accessing uninitialized data use EmbeddingVariable.

### **Graph & Grappler Optimization**

- Optimize Async EmbeddingLookup by placement optimization.
- Place VarHandlerOp to Compute main graph for SmartStage.
- Support independent thread pool for stage subgraph to avoid thread contention.
- Implement device placement optimization.

### **Runtime Optimization**

- Support CUDA Graph execution by adding CUDA Graph mode session.
- Support CUDA Graph execution in JIT mode.
- Support intra task cost estimate in CostModel in Executor.
- Support tf.stream and tf.colocate python API for CUDA multi-stream.
- Support embedding subgraphs partition policy when use CUDA multi-stream.
- Optimize CUDA multi-stream by merging copy stream into compute stream.

### **Ops & Hardware Acceleration**

- Add a list of Quantized* and _MklQuantized* ops.
- Implement GPU version of SparseFillEmptyRows.
- Implement c version of spin_lock to support multi-architectures.
- Upgrade the OneDNN version to v2.7.

### **Distributed**

- Support distributed training use SOK based on EmbeddingVariable.
- Add NETWORK_MAX_CONNECTION_TIMEOUT to support connection timeout configurable in StarServer.
- Upgrade the SOK version to v4.2.

### **IO**

- Add TF_NEED_PARQUET_DATASET to enable ParquetDataset.

### **Serving**

- Optimize embedding lookup performance by disable feature filter when serving.
- Optimize error code for user when parse request or response failed.
- Support independent update model threadpool to avoid performance jitter.

### **ModelZoo**

- Add MaskNet Model.
- Add PLE Model.
- Support variable type BF16 in DCN model.

### **BugFix**

- Fix tf.nn.embedding_lookup interface bug and session hang bug when enabling async embedding.
- Fix warmup failed bug when user set warmup file path.
- Fix build failure in ev_allocator.cc and hash.cc on ARM.
- Fix build failure in arrow when build on ARM
- Fix redefined error in NEON header file for ARM.
- Fix _mm_malloc build failure in sparsehash on ARM.
- Fix warmup failed bug when use session_group.
- Fix build save graph bug when creating partitioned EmbeddingVariable in feature_column API.
- Fix the colocation error when using EmbeddingVariable in distribution.
- Fix HostNameToIp fails by replacing gethostbyname by getaddrinfo in StarServer.

More details of features: [https://deeprec.readthedocs.io/zh/latest/](url)

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2210-cpu-py36-ubuntu18.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2210-gpu-py36-cu116-ubuntu18.04`

### **Thanks to our Contributors**
Duyi-Wang, Locke, shijieliu, Honglin Zhu, chenxujun, GosTraight2020, LALBJ, Nanno

# Release r1.15.5-deeprec2208u1

## **Major Features and Improvements**

### **BugFix**

- Fix a list of Quantized* and _MklQuantized* ops not found issue.
- Fix build save graph bug when creating partitioned EmbeddingVariable in feature_column API.
- Fix warmup failed bug when user set warmup file path.
- Fix warmup failed bug when use session_group.

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2208u1-cpu-py36-ubuntu18.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2208u1-gpu-py36-cu116-ubuntu18.04`

# Release r1.15.5-deeprec2208

## **Major Features and Improvements**

### **Embedding**

- Multi-tier of EmbeddingVariable support HBM, add async compactor in SSDHashKV.
- Support tf.feature_column.shard_embedding_columns, SequenceCategoricalColumn and WeightedCategoricalColumn API for EmbeddingVariable.
- Support save and restore checkpoint of GPU EmbeddingVariable.
- Support EmbeddingVariable OpKernel with REAL_NUMBER_TYPES.
- Support user defined default_value for feature filter.
- Support feature column API for MultiHash.

### **Graph & Grappler Optimization**

- Add FP32 fused l2 normalize op and grad op and tf.nn.fused_layer_normalize API.
- Add Concat+Cast fusion ops.
- Optimize SmartStage performance on GPU.
- Add macro to control to optimize mkl_layout_pass.
- Support asynchronous embedding lookup.

### **Runtime Optimization**

- CPUAllocator, avoid multiple threads cleanup at the same time.
- Support independent intra threadpool for each session and intra threadpool be pinned to cpuset.
- Support multi-stream with virtual device.

### **Ops & Hardware Acceleration**

- Implement ApplyFtrl, ResourceApplyFtrl, ApplyFtrlV2 and ResourceApplyFtrlV2 GPU kernels.
- Optimize BatchMatmul GPU kernel.
- Integrate cuBLASlt into backend and use BlasLtMatmul in batch_matmul_op.
- Support GPU fusion of matmal+bias+(activation).
- Merge NV-TF r1.15.5+22.06.

### **Optimizer**
- Support AdamW optimizer for EmbeddingVariable.

### **Model Save/Restore**

- Support asynchronously restore EmbeddingVariable from checkpoint.
- Support EmbeddingVariable in init_from_checkpoint.

### **Serving**

- Add go/java/python client SDK and demo.
- Support GPU multi-streams in SessionGroup.
- Support independent inter thread pool for each session in SessionGroup.
- Support multi-tiered Embedding.
- Support immutable EmbeddingVariable.

### **Quantization**

- Add low precision optimization tool, support BF16, FP16, INT8 for savedmodel and checkpoint.
- Add embedding variable quantization.

### **ModelZoo**

- Optimize DIN's BF16 performance.
- Add DCN & DCNv2 models and MLPerf recommendation benchmark.

### **Profiler**

- Add detail information for RecvTensor in timeline.

### **Dockerfile**

- Add ubuntu 22.04 dockerfile and images with gcc11.2 and python3.8.6.
- Add cuda11.2, cuda11.4, cuda11.6, cuda11.7 docker images and use cuda 11.6 as default GPU image.

### **Environment & Build**

- Update default TF_CUDA_COMPUTE_CAPABILITIES to 6.0,6.1,7.0,7.5,8.0.
- Upgrade bazel version to 0.26.1.
- Support for building DeepRec on ROCm2.10.0.

### **BugFix**

- Fix build failures with gcc11 & gcc12.
- StarServer, remove user packet split to avoid multiple user packet out-of-order issue.
- Fix the 'NodeIsInGpu is not declare' issue.
- Fix the placement bug of worker devices when distributed training in Modelzoo.
- Fix out of range issue for BiasAddGrad op when enable AVX512.
- Avoid loading invalid model when model update in serving.

More details of features: [https://deeprec.readthedocs.io/zh/latest/](url)

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2208-cpu-py36-ubuntu18.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2208-gpu-py36-cu116-ubuntu18.04`

# Release r1.15.5-deeprec2206

## **Major Features and Improvements**

### **Embedding**

- Multi-tier of EmbeddingVariable, add SSD_HashKV which is better performance than LevelDB.
- Support GPU EmbeddingVariable which gather/apply ops place on GPU.
- Add user API to record frequence and version for EmbeddingVariable.

### **Graph Optimization**

- Add Embedding Fusion ops for CPU/GPU.
- Optimize SmartStage performance on GPU.

### **Runtime Optimization**

- Executor, support cost-based and critical path ops first.
- GPUAllocator, support CUDA malloc async allocator.  (need to use >= CUDA 11.2)
- CPUAllocator, automatically memory allocation policy generation.
- PMEMAllocator, optimize allocator and add statistic.

### **Ops & Hardware Acceleration**

- Implement SparseReshape, SparseApplyAdam, SparseApplyAdagrad, SparseApplyFtrl, ApplyAdamAsync, SparseApplyAdamAsync, KvSparseApplyAdamAsync GPU kernels.
- Optimize UnSortedSegment on CPU.
- Upgrade OneDNN to v2.6.

### **IO & Dataset**

- ParquetDataset, add parquet dataset which could reduce storage and improve performance.

### **Model Save/Restore**

- Asynchronous restore EmbeddingVariable from checkpoint.

### **Serving**

- SessionGroup, highly improve QPS and RT in inference.

### **ModelZoo**

- Add models SimpleMultiTask, ESSM, DBMTL, MMoE, BST.

### **Profiler**

- Support for mapping of operators and real thread ids in timeline.

### **BugFix**

- Fix EmbeddingVariable core when EmbeddingVariable only has primary embedding value.
- Fix abnormal behavior in L2-norm calculation.
- Fix save checkpoint issue when use LevelDB in EmbeddingVariable.
- Fix delete old checkpoint failure when use incremental checkpoint.
- Fix build failure with CUDA 11.6.

More details of features: [https://deeprec.readthedocs.io/zh/latest/](url)

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2206-cpu-py36-ubuntu18.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2206-gpu-py36-cu110-ubuntu18.04`

# Release r1.15.5-deeprec2204u1

## **Major Features and Improvements**

### **BugFix**

- Fix saving checkpoint issue when use EmbeddingVariable. (https://github.com/alibaba/DeepRec/issues/167)
- Fix inputs from different frames issue when use auto graph fusion. (https://github.com/alibaba/DeepRec/issues/144)
- Fix embedding_lookup_sparse graph issue.

## **Release Images**

### **CPU Image**

`alideeprec/deeprec-release:deeprec2204u1-cpu-py36-ubuntu18.04`

### **GPU Image**

`alideeprec/deeprec-release:deeprec2204u1-gpu-py36-cu110-ubuntu18.04`

# Release r1.15.5-deeprec2204

## **Major Features and Improvements**

### **Embedding**

- Support hybrid storage of EmbeddingVariable (DRAM, PMEM, LevelDB)
- Support memory-continuous storage of multi-slot EmbeddingVariable.
- Optimize beta1_power and beta2_power slots of EmbeddingVariable.
- Support restore frequency of features in EmbeddingVariable.

### **Distributed Training**

- Integrate SOK in DeepRec.

### **Graph Optimization**

- Auto Graph Fusion, support float32/int32/int64 type for select fusion.
- SmartStage, fix graph contains circle bug when enable SmartStage optimization.

### **Runtime Optimization**

- GPUTensorPoolAllocator, which reduce GPU memory usage and improve performance.
- PMEMAllocator, support allocation in persistent memory. 

### **Optimizer**

- Optimize AdamOptimizer performance.

### **Op & Hardware Acceleration**

- Change fused MatMul layout type and number thread for small size inputs.

### **IO & Dataset**

- KafkaGroupIODataset, support consumer rebalance.

### **Model Save/Restore**

- Support dump incremental graph info.

### **Serving**

- Add serving module (ODL processor), which support Online Deep Learning (ODL).

More details of features: [https://deeprec.readthedocs.io/zh/latest/](url)

## **Release Images**

### **CPU Image**

`registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-training:deeprec2204-cpu-py36-ubuntu18.04`

### **GPU Image**

`registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-training:deeprec2204-gpu-py36-cu110-ubuntu18.04`

### Known Issue
Some user report issue when use Embedding Variable, such as https://github.com/alibaba/DeepRec/issues/167. The bug is fixed in r1.15.5-deeprec2204u1.

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

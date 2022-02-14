# Benchmarking Embedding Lookup Operations within an End-to-End Model Training

This repository contains a truncated version of DeepFM model implemented by TensorFlow (1.x). 
We keep the I/O and embedding lookup stage while removing the dense layers. This benchmark aims to 
evaluate how memory module (e.g., DRAM, Persistent memory) affect the training performance (throughput).

## Environment Setup

Please use a docker image of Aliyun PAI-DLC to run this benchmark. 
```bash
docker pull registry.cn-shanghai.aliyuncs.com/pai-dlc/tensorflow-training:1.15deeprec2106-gpu-py36-cu110-ubuntu18.04
```

Other required pip whls
```bash
numpy                         1.18.5
pandas                        1.1.5
```

## How to run the benchmark

```bash
## use mock data
./launch.sh \
  --use_mock_data=True \
  --num_mock_cols=30
  --batch_size=12800 \
  --dim_size=256 \
  --ev_storage=pmem_libpmem \

## use criteo data
./launch.sh \
  --use_mock_data=False \
  --data_dir=${path_to_criteo} \
  --batch_size=12800 \
  --dim_size=256 \
  --ev_storage=pmem_libpmem \
```

- Here `num_mock_cols` is the number of columns in mock training data, `mock_vocabulary_size` is 
the number of buckets for each column (variable) in mock data. `vocabulary_amplify_factor` is 
utilized to control the memory footprint in benchmarking. For instance a `vocabulary_amplify_factor=2`
would increase the `vocabulary_size` of each column (variable) by a factor of 2.
- The log files are located at `./bench-ps.log` and `./bench-worker.log` for ps and worker process, respectively.
- The default embedding variable type is hash table-based Embedding Variable, which does not require a specification on `mock_vocabulary_size` and `vocabulary_amplify_factor`
- by specifying `--use_ev_var=False`, the benchmark would fallback to use TF's native variable for Embeddings, and users shall provide `mock_vocabulary_size` in mock data scenarios.
- '--ev_storage=pmem_libpmem/pmem_memkind/dram' to select EmbeddingVariable StorageType, Default is 'dram'. If set pmem_libpmem, please also set pmem path and size with '--ev_storage_path=<pmem_path (default: /mnt/pmem0/allocator/)> and --ev_storage_size_gb=<pmem_size (default: 512)>'


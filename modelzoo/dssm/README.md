# DSSM
The following is a brief directory structure and description for this example:
```
├── data                          # Data set directory
│   └── README.md                   # Documentation describing how to prepare dataset
├── distribute_k8s                # Distributed training related files
│   ├── distribute_k8s_BF16.yaml    # k8s yaml to crate a training job with BF16 feature
│   ├── distribute_k8s_FP32.yaml    # k8s yaml to crate a training job
│   └── launch.py                   # Script to set env for distributed training
├── README.md                     # Documentation
├── result                        # Output directory
│   └── README.md                   # Documentation describing output directory
└── train.py                      # Training script
```

- [DSSM](#dssm)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
    - [Distribute Training](#distribute-training)
  - [Benchmark](#benchmark)
    - [Stand-alone Training](#stand-alone-training-1)
      - [Test Environment](#test-environment)
      - [Performance Result](#performance-result)
    - [Distributed Training](#distributed-training)
      - [Test Environment](#test-environment-1)
      - [Performance Result](#performance-result-1)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Fields](#fields)
    - [Processing](#processing)

## Model Structure
Deep Structured Semantic Model(DSSM) is a semantic model based on deep network to figure out the similarity between query and documents through mapping into the semantic space of common dimensions.  
The structure of model is as follow：
```
output:
                                   probability of a click
model:
                                              /|\
                                               |
                     _______________>  Cosine similarity  <______________
                    /                                                    \ 
                   |                                                      |
                   |                                                      |
                [Norm]                                                  [Norm]
           ________|________                                      ________|________ 
          |                 |                                    |                 |  
          |                 |                                    |                 |
          |                 |                                    |                 |
          |       DNN       |                                    |       DNN       |
          |                 |                                    |                 |
          |                 |                                    |                 |
          |_________________|                                    |_________________|
                   |                                                      |
            |_Emb_|____|__|                                        |_Emb_|____|__|
                   |                                                      |
input: 
             [user features]                                       [item features]
            
```


## Usage
### Stand-alone Training
1.  Please prepare the data set and DeepRec env.
    1.  Manually
        - Follow [dataset preparation](#prepare) to prepare data set.
        - Download code by `git clone https://github.com/alibaba/DeepRec`
        - Follow [How to Build](https://github.com/alibaba/DeepRec#how-to-build) to build DeepRec whl package and install by `pip install $DEEPREC_WHL`.
    2.  *Docker(Recommended)*
        ```
        docker pull alideeprec/deeprec-release-modelzoo:latest
        docker run -it alideeprec/deeprec-release-modelzoo:latest /bin/bash

        # In docker container
        cd /root/modelzoo/dssm
        ```

2.  Training.  
    ```
    python train.py
    
    # Memory acceleration with jemalloc.
    # The required ENV `MALLOC_CONF` is already set in the code.
    LD_PRELOAD=./libjemalloc.so.2.5.1 python train.py
    ```
    Use argument `--bf16` to enable DeepRec BF16 feature.
    ```
    python train.py --bf16

    # Memory acceleration with jemalloc.
    # The required ENV `MALLOC_CONF` is already set in the code.
    LD_PRELOAD=./libjemalloc.so.2.5.1 python train.py --bf16
    ```
    In the community tensorflow environment, use argument `--tf` to disable all of DeepRec's feature.
    ```
    python train.py --tf
    ```
    Use arguments to set up a custom configuation:
    - DeepRec Features:
      - `export START_STATISTIC_STEP` and `export STOP_STATISTIC_STEP`: Set ENV to configure CPU memory optimization. This is already set to 100 & 110 in the code by default.
      - `--bf16`: Enable DeepRec BF16 feature in DeepRec. Use FP32 by default.
      - `--emb_fusion`: Whether to enable embedding fusion, Default to True.
      - `--op_fusion`: Whether to enable Auto graph fusion feature. Default to True.
      - `--optimizer`: Choose the optimizer for deep model from ['adam', 'adamasync', 'adagraddecay', 'adagrad']. Use adagrad by default.
      - `--smartstaged`: Whether to enable smart staged feature of DeepRec, Default to True.
      - `--micro_batch`: Set num for Auto Mirco Batch. Default 0 to close.(Not really enabled)
      - `--ev`: Whether to enable DeepRec EmbeddingVariable. Default to False.
      - `--adaptive_emb`: Whether to enable Adaptive Embedding. Default to False.
      - `--ev_elimination`: Set Feature Elimination of EmbeddingVariable Feature. Options [None, 'l2', 'gstep'], default to None.
      - `--ev_filter`: Set Feature Filter of EmbeddingVariable Feature. Options [None, 'counter', 'cbf'], default to None.
      - `--dynamic_ev`: Whether to enable Dynamic-dimension Embedding Variable. Default to False.(Not really enabled)
      - `--incremental_ckpt`: Set time of save Incremental Checkpoint. Default 0 to close.
      - `--workqueue`: Whether to enable Work Queue. Default to False.
      - `--protocol`: Set the protocol ['grpc', 'grpc++', 'star_server'] used when starting server in distributed training. Default to grpc.
      - `--parquet_dataset`: Whether to enable ParquetDataset. Default is `True`. 
    - Basic Settings:
      - `--data_location`: Full path of train & eval data, default to `./data`.
      - `--steps`: Set the number of steps on train dataset. Default will be set to 1000 epoch.
      - `--no_eval`: Do not evaluate trained model by eval dataset.
      - `--batch_size`: Batch size to train. Default to 4096.
      - `--output_dir`: Full path to output directory for logs and saved model, default to `./result`.
      - `--checkpoint`: Full path to checkpoints input/output directory, default to `$(OUTPUT_DIR)/model_$(MODEL_NAME)_$(TIMESTAMPS)`
      - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to 0.
      - `--seed`: Set the random seed for tensorflow.
      - `--timeline`: Save steps of profile hooks to record timeline, zero to close, defualt to 0.
      - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default to 1.
      - `--inter`: Set inter op parallelism threads. Default to 0.
      - `--intra`: Set intra op parallelism threads. Default to 0.
      - `--input_layer_partitioner`: Slice size of input layer partitioner(units MB).
      - `--dense_layer_partitioner`: Slice size of dense layer partitioner(units kB).
      - `--tf`: Use TF 1.15.5 API and disable DeepRec features.


### Distribute Training
1. Prepare a K8S cluster. [Alibaba Cloud ACK Service(Alibaba Cloud Container Service for Kubernetes)](https://cn.aliyun.com/product/kubernetes) can quickly create a Kubernetes cluster. 
2. Perpare a shared storage volume. For Alibaba Cloud ACK, [OSS(Object Storage Service)](https://cn.aliyun.com/product/oss) can be used as a shared storage volume.
3. Create a PVC(PeritetVolumeClaim) named `deeprec` for storage volumn in cluster.
4. Prepare docker image. `alideeprec/deeprec-release-modelzoo:latest` is recommended.
5. Create a k8s job from `.yaml` to run distributed training.
   ```
   kubectl create -f $YAML_FILE
   ```
6. Show training log by `kubectl logs -f trainer-worker-0`


## Benchmark
### Stand-alone Training
#### Test Environment
The benchmark is performed on the [Alibaba Cloud ECS general purpose instance family with high clock speeds - **ecs.hfg7.2xlarge**](https://help.aliyun.com/document_detail/25378.html?spm=5176.2020520101.vmBInfo.instanceType.4a944df5PvCcED#hfg7).
- Hardware 
  - Model name:          Intel(R) Xeon(R) Platinum 8369HC CPU @ 3.30GHz
  - CPU(s):              8
  - Socket(s):           1
  - Core(s) per socket:  4
  - Thread(s) per core:  2
  - Memory:              32G

- Software
  - kernel:                 4.18.0-348.2.1.el8_5.x86_64
  - OS:                     CentOS Linux release 8.5.2111
  - GCC:                    8.5.0
  - Docker:                 20.10.12
  - Python:                 3.6.8

#### Performance Result

<table>
    <tr>
        <td colspan="1"></td>
        <td>Framework</td>
        <td>DType</td>
        <td>Accuracy</td>
        <td>AUC</td>
        <td>Throughput</td>
    </tr>
    <tr>
        <td rowspan="3">DSSM</td>
        <td>Community TensorFlow</td>
        <td>FP32</td>
        <td>0.913399</td>
        <td>0.503494</td>
        <td>75906.95(baseline)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32</td>
        <td>0.9079</td>
        <td>0.492405</td>
        <td>115607.72(1.52x)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32+BF16</td>
        <td>0.9265</td>
        <td>0.484328</td>
        <td>129099.08(1.70x)</td>
    </tr>
</table>

- Community TensorFlow version is v1.15.5.

### Distributed Training
#### Test Environment
The benchmark is performed on the [Alibaba Cloud ACK Service(Alibaba Cloud Container Service for Kubernetes)](https://cn.aliyun.com/product/kubernetes), the K8S cluster is composed of the following ten machines.

- Hardware 
  - Model name:          Intel(R) Xeon(R) Platinum 8369HC CPU @ 3.30GHz
  - CPU(s):              8
  - Socket(s):           1
  - Core(s) per socket:  4
  - Thread(s) per core:  2
  - Memory:              32G


#### Performance Result  

<table>
    <tr>
        <td colspan="1"></td>
        <td>Framework</td>
        <td>Protocol</td>
        <td>DType</td>
        <td>Throughput</td>
    </tr>
    <tr>
        <td rowspan="3">DSSM</td>
        <td>Community TensorFlow</td>
        <td>GRPC</td>
        <td>FP32</td>
        <td></td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>GRPC</td>
        <td>FP32</td>
        <td></td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>GRPC</td>
        <td>FP32+BF16</td>
        <td></td>
    </tr>
</table>

- Community TensorFlow version is v1.15.5.

## Dataset
Train & eval dataset using ***Taobao dataset***.
### Prepare
We provide the dataset in two formats:
1. **CSV Format**
Put data file **taobao_train_data & taobao_test_data** into ./data/    
These files are available at [Taobao CSV Dataset](https://github.com/AlibabaPAI/Modelzoo-Data/tree/main/csv/taobao).
2. **Parquet Format**
Put data file **taobao_train_data.parquet & taobao_test_data.parquet** into ./data/
These files are available at [Taobao Parquet Dataset](https://github.com/AlibabaPAI/Modelzoo-Data/tree/main/parquet_dataset/bst_taobao).


### Fields
The dataset contains 20 columns, details as follow:
| Name | clk      | buy      | pid       | adgroup_id | cate_id   | campaign_id | customer  | brand     | user_id   | cms_segid | cms_group_id | final_gender_code | age_level | pvalue_level | shopping_level | occupation | new_user_class_level | tag_category_list | tag_brand_list | price    |
| ---- | -------- | -------- | --------- | ---------- | --------- | ----------- | --------- | --------- | --------- | --------- | ------------ | ----------------- | --------- | ------------ | -------------- | ---------- | -------------------- | ----------------- | -------------- | -------- |
| Type | tf.int32 | tf.int32 | tf.string | tf.string  | tf.string | tf.string   | tf.string | tf.string | tf.string | tf.string | tf.string    | tf.string         | tf.string | tf.string    | tf.string      | tf.string  | tf.string            | tf.string         | tf.string      | tf.int32 |


The data in `tag_category_list` and `tag_brand_list` column are separated by `'|'` 

### Processing
The 'clk' column is used as labels.  
User's feature columns is as follow:
| Column name          | Hash bucket size | Embedding dimension |
| -------------------- | ---------------- | ------------------- |
| user_id              | 100000           | 16                  |
| cms_segid            | 100              | 16                  |
| cms_group_id         | 100              | 16                  |
| age_level            | 10               | 16                  |
| pvalue_level         | 10               | 16                  |
| shopping_level       | 10               | 16                  |
| occupation           | 10               | 16                  |
| new_user_class_level | 10               | 16                  |
| tag_category_list    | 100000           | 16                  |
| tag_brand_list       | 100000           | 16                  |

Item's feature columns is as follow:
| Column name | Hash bucket size | Embedding dimension |
| ----------- | ---------------- | ------------------- |
| pid         | 10               | 16                  |
| adgroup_id  | 100000           | 16                  |
| cate_id     | 10000            | 16                  |
| campaign_id | 100000           | 16                  |
| customer    | 100000           | 16                  |
| brand       | 100000           | 16                  |
| price       | 50               | 16                  |

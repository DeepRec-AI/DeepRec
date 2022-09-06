# MMOE

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

- [MMOE](#mmoe)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
    - [Distribute Training](#distribute-training)
  - [Benchmark](#benchmark)
    - [Stand-alone Training](#stand-alone-training-1)
      - [Test Environment](#test-environment)
      - [Performance Result](#performance-result)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Fields](#fields)
    - [Processing](#processing)
  - [TODO LIST](#todo-list)

## Model Structure
[Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)MMoE is Multi-Tower model in which each tower gets weighted inputs from multiple exprets
The model structure in this folder is as follow:
```
output:                Probability of click                      Probability of buy
                                ▲                                        ▲
                                │                                        │
model:                     ┌────┴────┐                              ┌────┴────┐
                           │         │                              │         │
                           │  click  │                              │   buy   │
                           │         │                              │         │
                           │   DNN   │                              │   DNN   │
                           │         │                              │         │
                           └─────────┘                              └─────────┘
                                ▲                                        ▲
                                │                                        │
                              ┌─┴─┐                                    ┌─┴─┐
             ┌───────────────►│ x │                                    │ x │◄───────────────┐
             │                └───┘                                    └───┘                │
             │                  ▲                                        ▲                  │
             │                  │                                        │                  │
             │                  ├────────────────────┬───────────────────┤                  │
             │                  │                    │                   │                  │
             │                  │                    │                   │                  │
             │             ┌────┴────┐          ┌────┴────┐         ┌────┴────┐             │
             │             │         │          │         │         │         │             │
             │             │         │          │         │         │         │             │
       ┌─────┴──────┐      │ Expert  │          │ Expert  │         │ Expert  │      ┌──────┴─────┐
       │            │      │         │          │         │         │         │      │            │
       │ click gate │      │    1    │          │   ...   │         │    N    │      │  buy gate  │
       │    DNN     │      │         │          │         │         │         │      │    DNN     │
       └────────────┘      │   DNN   │          │   DNN   │         │   DNN   │      └────────────┘
             ▲             │         │          │         │         │         │             ▲
             │             │         │          │         │         │         │             │
             │             └─────────┘          └─────────┘         └─────────┘             │
             │                  ▲                    ▲                   ▲                  │
             │                  │                    │                   │                  │
             │                  │                    │                   │                  │
             │                  └────────────────────┼───────────────────┘                  │
             │                                       │                                      │
             │                                       │                                      │
             │                                       │                                      │
             │                                  ┌────┴────┐                                 │
             │                                  │         │                                 │
             └──────────────────────────────────┤  input  ├─────────────────────────────────┘
                                                │         │
                                                └─────────┘
```
## Usage

### Stand-alone Training
1.  Please prepare the [data set](#prepare) first.

2.  Create a docker image by DockerFile.   
    Choose DockerFile corresponding to DeepRec(Pending) or Google tensorflow.
    ```
    docker build -t DeepRec_Model_Zoo_MMOE_training:v1.0 .
    ```

3.  Run a docker container.
    ```
    docker run -it DeepRec_Model_Zoo_MMOE_training:v1.0 /bin/bash
    ```

4.  Training.  
     ```
    cd /root/
    python train.py
    ```
    Use argument `--bf16` to enable DeepRec BF16 in deep model.
    ```
    python train.py --bf16
    ```
    Use arguments to set up a custom configuation:
    - DeepRec Features:
      - `export START_STATISTIC_STEP` and `export STOP_STATISTIC_STEP`: Set ENV to configure CPU memory optimization. This is already set to `100` & `110` in the code by default.
      - `--bf16`: Enable DeepRec BF16 feature in DeepRec. Use FP32 by default.
      - `--emb_fusion`: Whether to enable embedding fusion, Default is `True`.
      - `--op_fusion`: Whether to enable Auto graph fusion feature. Default is `True`.
      - `--optimizer`: Choose the optimizer for deep model from ['adam', 'adamasync', 'adagraddecay', 'adagrad', 'gradientdescent']. Use `adagrad` by default.
      - `--smartstaged`: Whether to enable SmartStaged feature of DeepRec, Default is `True`.
      - `--micro_batch`: Set num for Auto Micro Batch. Default is `0`. (Not really enabled)
      - `--ev`: Whether to enable DeepRec EmbeddingVariable. Default is `False`.
      - `--adaptive_emb`: Whether to enable Adaptive Embedding. Default is `False`.
      - `--ev_elimination`: Set Feature Elimination of EmbeddingVariable Feature. Options: [None, 'l2', 'gstep'], default is `None`.
      - `--ev_filter`: Set Feature Filter of EmbeddingVariable Feature. Options: [None, 'counter', 'cbf'], default to `None`.
      - `--dynamic_ev`: Whether to enable Dynamic-dimension Embedding Variable. Default is `False`. (Not really enabled)
      - `--multihash`: Whether to enable Multi-Hash Variable. Default is `False`. (Not really enabled)
      - `--incremental_ckpt`: Set time of save Incremental Checkpoint. Default is `0`.
      - `--workqueue`: Whether to enable WorkQueue. Default is `False`.
    - Basic Settings:
      - `--data_location`: Full path of train & eval data. Default is `./data`.
      - `--steps`: Set the number of steps on train dataset. When default(`0`) is used, the number of steps is computed based on dataset size and number of epochs equals 1000.
      - `--no_eval`: Do not evaluate trained model by eval dataset.
      - `--batch_size`: Batch size to train. Default is `512`.
      - `--output_dir`: Full path to output directory for logs and saved model. Default is `./result`.
      - `--checkpoint`: Full path to checkpoints output directory. Default is `$(OUTPUT_DIR)/model_$(MODEL_NAME)_$(TIMESTAMP)`
      - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to `None`.
      - `--seed`: Random seed. Default is `2021`.
      - `--timeline`: Save steps of profile hooks to record timeline, zero to close. Default is `None`.
      - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default is `1`.
      - `--learning_rate`: Learning rate for network. Default is `0.1`.
      - `--l2_regularization`: L2 regularization for the model. Default is `None`.
      - `--protocol`: Set the protocol('grpc', 'grpc++', 'star_server') used when starting server in distributed training. Default is `grpc`.
      - `--inter`: Set inter op parallelism threads. Default is `0`.
      - `--intra`: Set intra op parallelism threads. Default is `0`.
      - `--input_layer_partitioner`: Slice size of input layer partitioner(units MB). Default is `0`.
      - `--dense_layer_partitioner`: Slice size of dense layer partitioner(units kB). Default is `0`.
      - `--tf`: Use TF 1.15.5 API and disable all DeepRec features.


### Distribute Training
1. Prepare a K8S cluster and shared storage volume.
2. Create a PVC(PeritetVolumeClaim) for storage volumn in cluster.
3. Prepare docker image by DockerFile.
4. Edit k8s yaml file
- `replicas`: numbers of cheif, worker, ps.
- `image`: where nodes can pull the docker image.
- `claimName`: PVC name.

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
  - kernel:                 4.18.0-305.12.1.el8_4.x86_64
  - OS:                     CentOS Linux release 8.4.2105
  - Docker:                 20.10.12
  - Python:                 3.6.12

#### Performance Result
<table>
    <tr>
        <td colspan="1"></td>
        <td>Framework</td>
        <td>DType</td>
        <td>Accuracy</td>
        <td>AUC</td>
        <td>Globalsetp/Sec</td>
    </tr>
    <tr>
        <td rowspan="3">MMOE</td>
        <td>Community TensorFlow</td>
        <td>FP32</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32+BF16</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>

- Community TensorFlow version is v1.15.5.

## Dataset
Taobao dataset from [EasyRec](https://github.com/AlibabaPAI/EasyRec) is used.
### Prepare
Put data file **taobao_train_data & taobao_test_data** into ./data/    
For details of Data download, see [EasyRec](https://github.com/AlibabaPAI/EasyRec/#GetStarted)

### Fields
The dataset contains 20 columns, details as follow:
| Name | clk      | buy      | pid       | adgroup_id | cate_id   | campaign_id | customer  | brand     | user_id   | cms_segid | cms_group_id | final_gender_code | age_level | pvalue_level | shopping_level | occupation | new_user_class_level | tag_category_list | tag_brand_list | price    |
| ---- | -------- | -------- | --------- | ---------- | --------- | ----------- | --------- | --------- | --------- | --------- | ------------ | ----------------- | --------- | ------------ | -------------- | ---------- | -------------------- | ----------------- | -------------- | -------- |
| Type | tf.int32 | tf.int32 | tf.string | tf.string  | tf.string | tf.string   | tf.string | tf.string | tf.string | tf.string | tf.string    | tf.string         | tf.string | tf.string    | tf.string      | tf.string  | tf.string            | tf.string         | tf.string      | tf.int32 |


The data in `tag_category_list` and `tag_brand_list` column are separated by `'|'` 

### Processing
The 'clk' ans 'buy' columns are` used as labels.  
Input feature columns are as follow:
| Column name          | Hash bucket size | Embedding dimension |
| -------------------- | ---------------- | ------------------- |
| pid                  | 10               | 16                  |
| adgroup_id           | 100000           | 16                  |
| cate_id              | 10000            | 16                  |
| campaign_id          | 100000           | 16                  |
| customer             | 100000           | 16                  |
| brand                | 100000           | 16                  |
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
| -------------------- | Num Buckets      | ------------------- |
| price                | 50               | 16                  |

## TODO LIST
- Distribute training model
- Benchmark
- DeepRec DockerFile
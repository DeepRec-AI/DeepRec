# DIN

The following is a brief directory structure and description for this example:
```
├── data                          # Data set directory
│   ├── prepare_data.sh             # Shell script to download and process dataset
│   └── README.md                   # Documentation describing how to prepare dataset
│   └── script                      # Directory contains scripts to process dataset
│       ├── generate_voc.py           # Create a list of features
│       ├── local_aggretor.py         # Generate sample data
│       ├── pick2txt.py               # Convert voc's format
│       ├── process_data.py           # Parse raw json data
│       └── split_by_user.py          # Divide the dataset
├── distribute_k8s                # Distributed training related files
│   ├── distribute_k8s_BF16.yaml    # k8s yaml to crate a training job with BF16 feature
│   ├── distribute_k8s_FP32.yaml    # k8s yaml to crate a training job
│   └── launch.py                   # Script to set env for distributed training
├── README.md                     # Documentation
├── result                        # Output directory
│   └── README.md                   # Documentation describing output directory
└── train.py                      # Training script
```

## Content
- [DIN](#din)
  - [Content](#content)
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
[Deep Interest Network](https://arxiv.org/abs/1706.06978)(DIN) is proposed by Alibaba in 2017.6 which is a click-through rate(CRT) prediction model for e-commerce industry, focusing on mining the information in the user's historical behavior data.   

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
        cd /root/modelzoo/DIN
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
      - `--optimizer`: Choose the optimizer for deep model from ['adam', 'adamasync', 'adagraddecay']. Use adagrad by default.
      - `--smartstaged`: Whether to enable smart staged feature of DeepRec, Default to True.
      - `--micro_batch`: Set num for Auto Mirco Batch. Default 0 to close.(Not really enabled)
      - `--ev`: Whether to enable DeepRec EmbeddingVariable. Default to False.
      - `--adaptive_emb`: Whether to enable Adaptive Embedding. Default to False.
      - `--ev_elimination`: Set Feature Elimination of EmbeddingVariable Feature. Options [None, 'l2', 'gstep'], default to None.
      - `--ev_filter`: Set Feature Filter of EmbeddingVariable Feature. Options [None, 'counter', 'cbf'], default to None.
      - `--dynamic_ev`: Whether to enable Dynamic-dimension Embedding Variable. Default to False.(Not really enabled)
      - `--multihash`: Whether to enable Multi-Hash Variable. Default to False.(Not really enabled)
      - `--incremental_ckpt`: Set time of save Incremental Checkpoint. Default 0 to close.
      - `--workqueue`: Whether to enable Work Queue. Default to False.
      - `--protocol`: Set the protocol ['grpc', 'grpc++', 'star_server'] used when starting server in distributed training. Default to grpc. 
    - Basic Settings:
      - `--data_location`: Full path of train & eval data, default to `./data`.
      - `--steps`: Set the number of steps on train dataset. Default will be set to 1 epoch.
      - `--no_eval`: Do not evaluate trained model by eval dataset.
      - `--batch_size`: Batch size to train. Default to 512.
      - `--output_dir`: Full path to output directory for logs and saved model, default to `./result`.
      - `--checkpoint`: Full path to checkpoints input/output directory, default to `$(OUTPUT_DIR)/model_$(MODEL_NAME)_$(TIMESTAMPS)`
      - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to 0.
      - `--seed`: Set the random seed for tensorflow.
      - `--timeline`: Save steps of profile hooks to record timeline, zero to close, defualt to 0.
      - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default to 1.
      - `--learning_rate`: Learning rate for model. Default to 0.001.
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
        <td rowspan="3">DIN</td>
        <td>Community TensorFlow</td>
        <td>FP32</td>
        <td>0.626609</td>
        <td>0.655082</td>
        <td>10403.01(baseline)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32</td>
        <td>0.664656</td>
        <td>0.703606</td>
        <td>22422.64(2.16x)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32+BF16</td>
        <td>0.665547</td>
        <td>0.701519</td>
        <td></td>
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
        <td rowspan="3">DIN</td>
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
Amazon Dataset Books dataset is used as benchmark dataset.
### Prepare
Put data file into ./data/    
For details of Data download, see [Data Preparation](data/README.md)

### Fields
- cat_voc.pkl: Contain a list of book categories.
- mid_voc.pkl: Contain a list of item id(book id).
- uid_voc.pkl: Contain a list of user id.
- reviews-info: Contain a list of user's review.  
Each piece of data is as: `<user id>  <item id> <rating score> <timestamp>`
- item-info: Contain mapping relationship between item id and categories.  
`<item id>  <categories>`
- local_train_splitByUser & local_test_splitByUser: Train and evaluate dataset, consist of user id, item info, user's historical behavior.  
Each piece of data is as: `<label>  <user id>  <item id>  <categories>  <history item id list>  <history item categories list>`  
The history data are splitted by `''`

### Processing
Reviews are regard as behaviors and those from one user are sort by time. Assuming user *u* has *T* behaviors, the first *T-1* behaviors are used to predict whether user *u* will write the *T*-th review. 

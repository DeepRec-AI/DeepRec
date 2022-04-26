# WDL

- [WDL](#wdl)
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

[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)(WDL) is proposed by Google in 2016.   


## Model Structure
The WDL model structure & code in this repo refer to [Intel model zoo](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds).  
The hide units of DNN network is [1024, 512, 256]. There is a difference between this and Intel version on data processing. Continuous columns input as numeric column after normalization, expect "I10" that input as identity column, and categorical column input as embedding column after hashed. For details of data procesing, see [Dataset Processing](#processing).

The model structure is as follow:  
The input of model is consist of dense features and spare features.
The former is a vector of floating-point numbers, and the latter is a list of sparse indices.
The model is divided into two parts, Linear model and DNN model.
Linear model take the combine of dense features and sparse features as input,
while DNN model take the combine of dense features and the embedding table of sparse feature as input.
The model's output is the probability of a click calculated by the output of Linear and DNN model.
```
output:
                                   probability of a click
model:
                                              /|\
                                               |
                      _____________________>  ADD  <______________________
                    /                                                      \ 
                    |                                              ________|________ 
                    |                                             |                 |
                    |                                             |                 |
                    |                                             |                 |
                Linear Op                                         |       DNN       |
                    /\                                            |                 |
                   /__\                                           |                 |
                    |                                             |_________________|
                    |                                                      /\
                    |                                                     /__\
                    |                                                   ____|_____
                    |                                                 /            \
                    |                                                /       |_Emb_|____|__|
                    |                                               |               |
    [dense features, sparse features]                       [dense features] [sparse features]
                    |_______________________________________________________|
input:                                          |
                                 [dense features, sparse features]
```
## Usage

### Stand-alone Training
1.  Please prepare the [data set](#prepare) first.

2.  Create a docker image by DockerFile.   
    Choose DockerFile corresponding to DeepRec(Pending) or Google tensorflow.
    ```
    docker build -t DeepRec_Model_Zoo_WDL_training:v1.0 .
    ```

3.  Run a docker container.
    ```
    docker run -it DeepRec_Model_Zoo_WDL_training:v1.0 /bin/bash
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
    - `--data_location`: Full path of train & eval data, default to `./data`.
    - `--output_dir`: Full path to output directory for logs and saved model, default to `./result`.
    - `--checkpoint`: Full path to checkpoints input/output directory, default to `$(OUTPUT_DIR)/model_$(MODEL_NAME)_$(TIMESTAMPS)`
    - `--steps`: Set the number of steps on train dataset. Default will be set to 10 epoch.
    - `--batch_size`: Batch size to train. Default to 512.
    - `--timeline`: Save steps of profile hooks to record timeline, zero to close, defualt to 0.
    - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to 0.
    - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default to 1.
    - `--deep_learning_rate`: Learning rate for deep network. Default to 0.05.
    - `--linear_learning_rate`: Learning rate for linear model. Default to 0.2.
    - `--bf16`: Enable DeepRec BF16 feature in DeepRec. Use FP32 by default.
    - `--no_eval`: Do not evaluate trained model by eval dataset.
    - `--inter`: Set inter op parallelism threads. Default to 0.
    - `--intra`: Set intra op parallelism threads. Default to 0.
    - `--input_layer_partitioner`: Slice size of input layer partitioner(units MB).
    - `--dense_layer_partitioner`: Slice size of dense layer partitioner(units kB).
    - `--protocol`: Set the protocol("grpc", "grpc++", "star_server") used when starting server in distributed training. Default to grpc. 

### Distribute Training
1. Prepare a K8S cluster and shared storage volume.
2. Create a PVC(PeritetVolumeClaim) for storage volumn in cluster.
3. Prepare docker image by DockerFile.
4. Edit k8s yaml file
- `replicas`: numbers of cheif, worker, ps.
- `image`: where nodes can pull the docker image.
- `claimName`: PVC name.

### Use GPU
  In an environment with NVIDIA GPUs, the EmbeddingVaribles along with other Ops that have GPU support will automatically placed on GPU.
  Since the current implementation of GPU EmbeddingVariable cannot work with TensorFlow Saver, we need run with:
  ```
  python train.py --no_saver
  ```
  (Note: Since there are some Ops without GPU supporting in this model, the performance is not good with GPUs.)

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
  - GCC:                    8.4.1
  - Docker:                 20.10.9
  - Python:                 3.6.8

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
        <td rowspan="3">WDL</td>
        <td>Community TensorFlow</td>
        <td>FP32</td>
        <td>0.7671165</td>
        <td>0.7502928</td>
        <td>31.1121 (baseline)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32</td>
        <td>0.7664720</td>
        <td>0.7505082</td>
        <td>31.1755 (+1.00x)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32+BF16</td>
        <td>0.7690585</td>
        <td>0.7568862</td>
        <td>44.3273 (+1.42x)</td>
    </tr>
</table>

- Community TensorFlow version is v1.15.5.

## Dataset
Train & eval dataset using ***Kaggle Display Advertising Challenge Dataset (Criteo Dataset)***.
### Prepare
Put data file **train.csv & eval.csv** into ./data/    
For details of Data download, see [Data Preparation](data/README.md)

### Fields
Total 40 columns:  
**[0]:Label** - Target variable that indicates if an ad was clicked or not(1 or 0)  
**[1-13]:I1-I13** - A total 13 columns of integer continuous features(mostly count features)  
**[14-39]:C1-C26** - A total 26 columns of categorical features. The values have been hashed onto 32 bits for anonymization purposes.

Integer column's distribution is as follow:
| Column | 1    | 2     | 3     | 4   | 5       | 6      | 7     | 8    | 9     | 10  | 11  | 12   | 13   |
| ------ | ---- | ----- | ----- | --- | ------- | ------ | ----- | ---- | ----- | --- | --- | ---- | ---- |
| Min    | 0    | -3    | 0     | 0   | 0       | 0      | 0     | 0    | 0     | 0   | 0   | 0    | 0    |
| Max    | 1539 | 22066 | 65535 | 561 | 2655388 | 233523 | 26279 | 5106 | 24376 | 9   | 181 | 1807 | 6879 |

Categorical column's numbers of types is as follow:
| column | C1   | C2  | C3      | C4     | C5  | C6  | C7    | C8  | C9  | C10   | C11  | C12     | C13  | C14 | C15   | C16     | C17 | C18  | C19  | C20 | C21     | C22 | C23 | C24    | C25 | C26   |
| ------ | ---- | --- | ------- | ------ | --- | --- | ----- | --- | --- | ----- | ---- | ------- | ---- | --- | ----- | ------- | --- | ---- | ---- | --- | ------- | --- | --- | ------ | --- | ----- |
| nums   | 1396 | 553 | 2594031 | 698469 | 290 | 23  | 12048 | 608 | 3   | 65156 | 5309 | 2186509 | 3128 | 26  | 12750 | 1537323 | 10  | 5002 | 2118 | 4   | 1902327 | 17  | 15  | 135790 | 94  | 84305 |

### Processing
- Interger columns **I[1-9,11-13]** is processed with `tf.feature_column.numeric_column()` function, and the data is normalized.  
    In order to save time, the data required for normalization has been calculated in advance.
- Interger columns **I10** is processed with `tf.feature_column.categorical_column_with_identity()` function, and then packed by ```tf.feature_column.indicator_column()``` fucntion.
- Categorical columns **C[1-26]** is processed with `tf.feature_column.embedding_column()` function after using `tf.feature_column.categorical_column_with_hash_bucket()` function.

## TODO LIST
- Benchmark
- DeepRec DockerFile
# DLRM using SparseOperationKit #
- [DLRM](#dlrm)
  - [Model Structure](#model-structure)
  - [Training](#training)
    - [Prepare dataset](#prepare-dataset)
    - [Create environment](#create-environment)
    - [Train](#train)
  - [Benchmark](#benchmark)
    - [Test Environment](#test-environment)
    - [Stand-alone Training](#stand-alone-training)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Field](#field)
    - [Processing](#processing)


## Model Structure
[Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://github.com/facebookresearch/dlrm)(DLRM) is proposed by Facebook.  
The model structure in this folder is as follow:
```
output:
                  probability of a click
model:                      |
                           /\
                          /__\
                            |
     ___________________> Dot  <___________________
    /                       |                      \
   /\                       |                       |
  /__\                 ____/__\_____           ____/__\____
   |                  |_Emb_|____|__|    ...  |_Emb_|__|___|
input:
[ dense features ]     [sparse indices] , ..., [sparse indices]
```
The triangles represent mlp network. The inputs consists of dense features and sparse features. The former is a vector of floating point values and the latter is a list of sparse indices. After processing by botton mlp network, dense features are entered into a dot operation with the embedding of sparse features. The result of dot is input into top mlp network to get the prediction of a click.

## Training
### Prepare dataset

Please prepare the [dataset](#prepare) first.

### Create environment

Run a docker container and install SOK as the [doc](../../sparse_operation_kit/ReadMe.md)

### Train  
#### Set common params ###
```shell
$ export EMBEDDING_DIM=128
```

#### Run DLRM with TensorFlow 

```shell
$  export SLURM_TASKS_PER_NODE=8
$  horovodrun -np 8 ./wrapper.sh python3 train_stand.py \
    --global_batch_size=65536 \
    --train_file_pattern="./train/" \
    --test_file_pattern="./test/" \
    --embedding_layer="TF" \
    --embedding_vec_size=$EMBEDDING_DIM \
    --bottom_stack 512 256 $EMBEDDING_DIM \
    --top_stack 1024 1024 512 256 1 \
    --distributed_tool="horovod" 
```
#### Run DLRM with SOK
    
```shell
$  export SLURM_TASKS_PER_NODE=8
$  horovodrun -np 8 ./wrapper.sh python3 train_stand.py \
    --global_batch_size=8192 \
    --train_file_pattern="./train/" \
    --test_file_pattern="./test/" \
    --embedding_layer="SOK" \
    --embedding_vec_size=$EMBEDDING_DIM \
    --bottom_stack 512 256 $EMBEDDING_DIM \
    --top_stack 1024 1024 512 256 1 \
    --distributed_tool="horovod" 
```
    ```
    Use arguments to set up a custom configuation:
    - `--global_batch_size`: Batch size to train. 
    - `--train_file_pattern`: Full path of train data.
    - `--test_file_pattern`: Full path of eval data.
    - `--embedding_layer`: Type of embedding layer, choices=["TF", "SOK"].
    - `--embedding_vec_size`: Embedding dim.
    - `--bottom_stack`: The shape of dense layer in bottom MLP.
    - `--top_stack`: The shape of dense layer in top MLP.
    - `--distribute_strategy`: choices=["onedevice", "horovod"]

## Benchmark
### Test Environment
- Hardware 
  - CPU: 2x 64-Core (4-Die) AMD EPYC 7742 (-MT MCP MCM SMP-)          
  - GPU: 8x NVIDIA A100-SXM4-80GB

- Software
  - Driver:470.82.01      

### Stand-alone Training 
Google tensorflow v1.15 is selected to compare with SOK.

| environment | Exit criteria | embedding  | Device   | embedding vector size  | vocabulary size in bytes | Batch size(global) | Dataloader    | Data format | Averge time of iteration(ms) | Total time(minutes) |     |
|-------------|---------------|------------|----------|------------------------|--------------------------|--------------------|---------------|-------------|------------------------------|---------------------|-----|
| DeepRec     | 1 epoch       | SOK(GPU)   | 8 * A100 | 128                    | 82GB                     | 65536              | BinaryDataset | bin         | 8.66                         | 9.35                | 1 x |
| DeepRec     | 1 epoch       | TF(GPU)    | 8 * A100 | 128                    | 82GB                     | 65536              | BinaryDataset | bin         | OOM                          | OOM                 | -   |
| DeepRec     | 1 epoch       | TF(CPU)    | 8 * A100 | 128                    | 82GB                     | 65536              | BinaryDataset | bin         | -                            | -                   | -   |
| DeepRec     | 1 epoch       | TF_EV(CPU) | 8 * A100 | 128                    | -                        | 65536              | BinaryDataset | bin         | -                            | -                   | -   |


## Dataset
Train & eval dataset using Criteo TeraBytes Datasets.
### Prepare
Put data file **train.csv & eval.csv** into ./data/    
For details of Data download, see [Data Preparation](data/ReadMe.md)

### Field
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
| nums   | 39884406 | 39043 | 17289 | 7420 | 20263 | 3  | 7120 | 1543 | 63   | 38532951| 2953546 | 403346 | 10 | 2208  | 11938 | 155 | 4  | 976 | 14 | 39979771  | 25641295 | 39664984 | 585935  | 12972 | 108  | 36 |

### Processing
- Interger columns **I[1-13]** is processed with `tf.keras.layers.Dense()` function, and the data is normalized.  
    In order to save time, the data required for normalization has been calculated in advance.
- Categorical columns **C[1-26]** is processed with `tf.keras.layers.Embedding()` function for TensorFlow, and processed with `sok.All2AllDenseEmbedding()` for SOK.



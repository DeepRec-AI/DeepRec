# DLRM using SparseOperationKit #
- [DLRM](#dlrm)
  - [Model Structure](#model-structure)
  - [Training](#training)
    - [Prepare dataset](#prepare-dataset)
    - [Create environment](#create-environment)
    - [Train](#train)
  - [Benchmark](#benchmark)
    - [Test Environment](#test-environment)
    - [Stand-alone Training](#stand-alone-training-1)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Field](#field)
    - [Processing](#processing)
  - [TODO LIST](#todo-list)

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

Please prepare the [data set](#prepare) first.

### Create environment

Run a docker container and install SOK as the [doc](../../sparse_operation_kit/ReadMe.md)

### Train  
#### Set common params ###
```shell
$ export EMBEDDING_DIM=32
```
 
#### Run DLRM with TensorFlow

```shell
$  python3 train_stand.py \
    --global_batch_size=16384 \
    --train_file_pattern="./train/*.csv" \
    --test_file_pattern="./test/*.csv" \
    --embedding_layer="TF" \
    --embedding_vec_size=$EMBEDDING_DIM \
    --bottom_stack 512 256 $EMBEDDING_DIM \
    --top_stack 1024 1024 512 256 1 \
    --distributed_tool="onedevice" 
```
#### Run DLRM with SOK
    
```shell
$  python3 train_stand.py \
    --global_batch_size=16384 \
    --train_file_pattern="./train/*.csv" \
    --test_file_pattern="./test/*.csv" \
    --embedding_layer="SOK" \
    --embedding_vec_size=$EMBEDDING_DIM \
    --bottom_stack 512 256 $EMBEDDING_DIM \
    --top_stack 1024 1024 512 256 1 \
    --distributed_tool="onedevice"
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
  - CPU:                    
  - vCPU(s):               
  - Socket(s):              
  - Core(s) per socket:     
  - Thread(s) per core:     
  - Memory:                
  - L1d cache:            
  - L1i cache:             
  - L2 cache:               
  - L3 cache:               

- Software
  - kernel:                 
  - OS:                     
  - GCC:                    
  - Docker:                 
  - Python:                 

### Stand-alone Training 
Google tensorflow v1.15 is selected to compare with SOK.

<table>
    <tr>
        <td colspan="2"></td>
        <td>Accuracy</td>
        <td>AUC</td>
        <td>Globalsetp/Sec</td>
    </tr>
    <tr>
        <td rowspan="2">DLRM</td>
        <td>google TF FP32</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>SOK FP32</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>

## Dataset
Train & eval dataset using Criteo TeraBytes Datasets.
### Prepare
Put data file **train.csv & eval.csv** into ./data/    
For details of Data download, see [Data Preparation](data/ReadMe.md)

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
| nums   | 39884407 | 39043 | 17289 | 7420 | 20263 | 3  | 7120 | 1543 | 63   | 38532952 | 2953546 | 403346 | 10 | 2208  | 11938 | 155 | 4  | 976 | 14 | 39979772   | 25641295 | 39664985  | 585935  | 12972 | 108  | 36 |

### Processing
- Interger columns **I[1-13]** is processed with `tf.feature_column.numeric_column()` function, and the data is normalized.  
    In order to save time, the data required for normalization has been calculated in advance.
- Categorical columns **C[1-26]** is processed with `tf.keras.layers.Embedding()` function for TensorFlow, and processed with `sok.All2AllDenseEmbedding()` for SOK.

## TODO LIST
- Benchmark

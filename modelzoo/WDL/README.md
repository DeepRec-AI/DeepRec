# WDL

- [WDL](#wdl)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Fields](#fields)
    - [Processing](#processing)
  - [Benchmark](#benchmark)
    - [Test Environment](#test-environment)
    - [Stand-alone training](#stand-alone-training-1)

[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)(WDL) is proposed by Google in 2016.   


## Model Structure
The WDL model structure & code in this repo refer to [Intel model zoo](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds).  
In this folder, there are two different size WDL model as follow:  
- WDL_small: A small version WDL with model size of **1.8G**. The hide units of DNN network is [1024, 512, 256]. There is a difference between this and Intel version on data processing. Continuous columns input as numeric column after normalization, expect "I10" that input as identity column, and categorical column input as embedding column after hashed. For details of data procesing, see [Dataset Processing](#processing).
- WDL_large: A large version WDL with model size of **19G**. Large version sets hide units to [1536, 768, 64] and expands the hash bucket size of categorical to at least 2 times the data range.

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
    Default training small version.
    ```
    cd /home/
    python train_stand.py
    ```
    Use argument `--large_version` to train large version.
    ```
    python train_stand.py --large_version True
    ```
    Use argument `--bf16` to enable DeepRec BF16 in deep model.
    ```
    python train_stand.py --bf16 True
    ```
    Use arguments to set up a custom configuation:
    - `--data_location`: Full path of train & eval data, default is `./data`.
    - `--output_dir`: Full path to output directory for logs and saved model, default is `./result`.
    - `--steps`: Set the number of steps on train dataset. Default will be set to 10 epoch.
    - `--batch_size`: Batch size to train. Default is 512.
    - `--profile_steps`: Save steps of profile hooks to record timeline, zero to close, defualt is 0.
    - `--save_steps`: Set the number of steps on saving checkpoints. Default will be set to 500.
    - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default is 1.
    - `--deep_learning_rate`: Learning rate for deep network. Default is 0.05.
    - `--linear_learning_rate`: Learning rate for linear model. Default is 0.2.
    - `--bf16`: Enable DeepRec BF16 feature in deep model. Use FP32 by default.

## Dataset
Train & eval dataset using large Kaggle Display Advertising Challenge Dataset (Criteo Dataset).
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


## Benchmark
### Test Environment
The benchmark is performed on the Aliyun ECS.
- Hardware 
  - CPU:                    Intel(R) Xeon(R) Platinum 8369HB CPU @ 3.30GHz  
  - vCPU(s):                16
  - Socket(s):              1
  - Core(s) per socket:     8
  - Thread(s) per core:     2
  - Memory:                 64G  
  - L1d cache:              32K
  - L1i cache:              32K
  - L2 cache:               1024K
  - L3 cache:               33792K

- Software
  - kernel:                 4.18.0-305.3.1.el8.x86_64
  - OS:                     CentOS 8.4.2105
  - GCC:                    8.4.1
  - Docker:                 20.10.8
  - Python:                 3.6.9

### Stand-alone training 
Inter version benchmark please follow offical [training](https://github.com/IntelAI/models/blob/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds/training/fp32/README.md) usage.  
Google tensorflow v1.15 is selected to compare with DeepRec.
<!-- 
|               |              | Accuracy  | AUC        | Globalsetp/Sec |
|---------------|--------------|-----------|------------|----------------|
| Intel Version |              | 0.766265  | 0.7376816  | 50.45721134    |
| small version | google FP32  | 0.7671165 | 0.7502928  | 54.40165517    |
|               | DeepRec FP32 | 0.766472  | 0.7505082  | 51.8766        |
|               | DeepRec BF16 | 0.7690585 | 0.7568862  | 58.14365676    |
| large version | google FP32  | 0.7611555 | 0.74074477 | 42.15681935    |
|               | DeepRec FP32 | 0.768889  | 0.7413152  | 43.72933387    |
|               | DeepRec BF16 | 0.767139  | 0.7529076  | 47.12602419    | -->

<table>
    <tr>
        <td colspan="2"></td>
        <td>Accuracy</td>
        <td>AUC</td>
        <td>Globalsetp/Sec</td>
    </tr>
    <tr>
        <td colspan="2">Intel Version</td>
        <td>0.766265</td>
        <td>0.7376816</td>
        <td>50.45721134</td>
    </tr>
    <tr>
        <td rowspan="3">small version</td>
        <td>google FP32</td>
        <td>0.7671165</td>
        <td>0.7502928</td>
        <td>54.40165517</td>
    </tr>
    <tr>
        <td>DeepRec FP32</td>
        <td>0.766472</td>
        <td>0.7505082</td>
        <td>51.8766</td>
    </tr>
    <tr>
        <td>DeepRec BF16</td>
        <td>0.7690585</td>
        <td>0.7568862</td>
        <td>58.14365676</td>
    </tr>
    <tr>
        <td rowspan="3">large version</td>
        <td>google FP32</td>
        <td>0.7611555</td>
        <td>0.74074477</td>
        <td>42.15681935</td>
    </tr>
    <tr>
        <td>DeepRec FP32</td>
        <td>0.768889</td>
        <td>0.7413152</td>
        <td>43.72933387</td>
    </tr>
    <tr>
        <td>DeepRec BF16</td>
        <td>0.767139</td>
        <td>0.7529076</td>
        <td>47.12602419</td>
    </tr>
</table>
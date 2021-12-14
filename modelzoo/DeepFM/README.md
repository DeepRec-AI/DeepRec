# DeepFM

- [DeepFM](#deepfm)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
    - [Distribute Training](#distribute-training)
  - [Benchmark](#benchmark)
    - [Test Environment](#test-environment)
    - [Standing-alone training](#standing-alone-training)
    - [Distribute Training](#distribute-training-1)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Fields](#fields)
    - [Processing](#processing)
  - [TODO LIST](#todo-list)

## Model Structure
[DeepFM](https://arxiv.org/abs/1703.04247) is a CRT recommender model proposed in 2017 which combines the power of factorization machines for recommendation and deep learning for feature learning in a new neural network architecture. Compared to WDL model, wide and deep part of DeepFM share input so that feature engineering besides raw features is not needed.
The model's output is the probability of a click calculated by the output of FM and DNN model.
```
output:
                                   probability of a click
model:
                                              /|\
                                               |
                      _____________________>  ADD  <______________________
                    /                                                      \ 
             ________|________                                     ________|________ 
            |                 |                                   |                 |
            |                 |                                   |                 |
            |                 |                                   |                 |
            |       FM        |                                   |       DNN       |
            |                 |                                   |                 |
            |                 |                                   |                 |
            |_________________|                                   |_________________|
                    |                                                       |
                    |_______________________________________________________|
                                            ____|_____
                                          /            \
                                         /       |_Emb_|____|__|
                                        |               |
input:                                  |               |
                                 [dense features, sparse features]
```

## Usage
### Stand-alone Training
1.  Please prepare the [data set](#prepare) first.

2.  Create a docker image by DockerFile.   
    Choose DockerFile corresponding to DeepRec(Pending) or Google tensorflow.
    ```
    docker build -t DeepRec_Model_Zoo_DeepFM_training:v1.0 .
    ```

3.  Run a docker container.
    ```
    docker run -it DeepRec_Model_Zoo_DeepFM_training:v1.0 /bin/bash
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
    - `--steps`: Set the number of steps on train dataset. Default will be set to 1000 epoch.
    - `--batch_size`: Batch size to train. Default to 512.
    - `--timeline`: Save steps of profile hooks to record timeline, zero to close, defualt to 0.
    - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to 0.
    - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default to 1.
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

## Benchmark
The benchmark of example
### Test Environment
The information of hardware & software test environment

### Standing-alone training 
The benchmark of Standing-alone training 

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
        <td rowspan="3">DeepFM</td>
        <td>Community TensorFlow</td>
        <td>FP32</td>
        <td></td>
        <td></td>
        <td> (baseline)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32</td>
        <td></td>
        <td></td>
        <td> (+1.00x)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32+BF16</td>
        <td></td>
        <td></td>
        <td> (+1.00x)</td>
    </tr>
</table>

- Community TensorFlow version is v1.15.5.

### Distribute Training 
The benchmark of distribute training 

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
- Interger columns **I[1-13]** is processed with `tf.feature_column.numeric_column()` function, and the data is normalized.  
    In order to save time, the data required for normalization has been calculated in advance.
- Categorical columns **C[1-26]** is processed with `tf.feature_column.embedding_column()` function after using `tf.feature_column.categorical_column_with_hash_bucket()` function.



## TODO LIST
- Benchmark
- DeepRec DockerFile
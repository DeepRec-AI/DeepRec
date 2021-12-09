# DSSM

- [DSSM](#dssm)
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
1.  Please prepare the [data set](#prepare) first.

2.  Create a docker image by DockerFile.   
    Choose DockerFile corresponding to DeepRec(Pending) or Google tensorflow.
    ```
    docker build -t DeepRec_Model_Zoo_DSSM_training:v1.0 .
    ```

3.  Run a docker container.
    ```
    docker run -it DeepRec_Model_Zoo_DSSM_training:v1.0 /bin/bash
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
    - Swtich to enable DeepRec's feature:
      - `--bf16`: Enable BF16 feature in DeepRec. Use FP32 by default.
      - `--smartstaged`: Enable DeepRec smartstaged feature. Default to close.
      - `--protocol`: Set the protocol("grpc", "grpc++", "star_server") used when starting server in distributed training. Default to grpc. 
    - `--data_location`: Full path of train & eval data, default to `./data`.
    - `--output_dir`: Full path to output directory for logs and saved model, default to `./result`.
    - `--checkpoint`: Full path to checkpoints input/output directory, default to `$(OUTPUT_DIR)/model_$(MODEL_NAME)_$(TIMESTAMPS)`
    - `--steps`: Set the number of steps on train dataset. Default will be set to 1000 epoch.
    - `--batch_size`: Batch size to train. Default to 4096.
    - `--timeline`: Save steps of profile hooks to record timeline, zero to close, defualt to 0.
    - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to 0.
    - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default to 1.
    - `--no_eval`: Do not evaluate trained model by eval dataset.
    - `--inter`: Set inter op parallelism threads. Default to 0.
    - `--intra`: Set intra op parallelism threads. Default to 0.
    - `--input_layer_partitioner`: Slice size of input layer partitioner(units MB).
    - `--dense_layer_partitioner`: Slice size of dense layer partitioner(units kB).


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
The benchmark is performed on the [Alibaba Cloud ECS general purpose instance family with high clock speeds - **hfg7**](https://help.aliyun.com/document_detail/25378.html?spm=5176.2020520101.vmBInfo.instanceType.4a944df5PvCcED#hfg7).
- Hardware 
  - CPU:                    Intel(R) Xeon(R) Platinum 8369HB CPU @ 3.30GHz  
  - vCPU(s):                8
  - Socket(s):              1
  - Core(s) per socket:     4
  - Thread(s) per core:     2
  - Memory:                 32G  

- Software
  - kernel:                 4.18.0-305.3.1.el8.x86_64
  - OS:                     CentOS 8.4.2105
  - GCC:                    8.4.1
  - Docker:                 20.10.8
  - Python:                 3.6.9

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
        <td rowspan="3">DLRM</td>
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

- Community TensorFlow version is v1.15.

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


## TODO LIST
- Distribute training
- Benchmark
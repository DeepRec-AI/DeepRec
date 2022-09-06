# DIEN

- [DIEN](#dien)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
  - [Benchmark](#benchmark)
    - [Standing-alone training](#standing-alone-training)
      - [Test Environment](#test-environment)
      - [Performance Result](#performance-result)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Fields](#fields)
    - [Processing](#processing)
  - [TODO LIST](#todo-list)


## Model Structure
[Deep Interest Evolution Network](https://arxiv.org/abs/1809.03672)(DIEN) is proposed by Alibaba in 2018.11 which is a click-through rate(CRT) prediction model for e-commerce industry, focusing on capturing temporal interests from the user's historical behavior sequence.   

## Usage
### Stand-alone Training
1.  Please prepare the [data set](#prepare) first.
2.  Create a docker image by DockerFile.   
    Choose DockerFile corresponding to DeepRec(Pending) or Google tensorflow.
    ```
    docker build -t DeepRec_Model_Zoo_DIEN_training:v1.0 .
    ``` 
3.  Run a docker container.
    ```
    docker run -it DeepRec_Model_Zoo_DIEN_training:v1.0 /bin/bash
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
    - `--data_location`: Full path of train & eval data, default is `./data`.
    - `--output_dir`: Full path to output directory for logs and saved model, default is `./result`.
    - `--steps`: Set the number of steps on train dataset. Default will be set to 10 epoch.
    - `--batch_size`: Batch size to train. Default is 512.
    - `--timeline`: Save steps of profile hooks to record timeline, zero to close, defualt to 0.
    - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to 0.
    - `--bf16`: Enable DeepRec BF16 feature in DeepRec. Use FP32 by default.
    - `--no_eval`: Do not evaluate trained model by eval dataset.


## Benchmark
### Standing-alone training 
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

## TODO LIST
- Distributed training and benchmark

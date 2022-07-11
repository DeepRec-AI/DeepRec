# WDL

## Content
- [WDL](#wdl)
  - [Content](#content)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
  - [Dataset](#dataset)

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
        cd /root/modelzoo/WDL
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
      - `--learning_rate`: Learning rate for model.
      - `--tf`: Use TF 1.15.5 API and disable DeepRec features.

## Dataset
Train & eval dataset using [***Ali ELM dataset***](https://tianchi.aliyun.com/dataset/dataDetail?dataId=131047).

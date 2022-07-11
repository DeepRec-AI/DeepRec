# DeepFM

## Content
- [DeepFM](#deepfm)
  - [Content](#content)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
  - [Dataset](#dataset)

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
        cd /root/modelzoo/DeepFM
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
      - `--optimizer`: Choose the optimizer for deep model from ['adam', 'adamasync', 'adagraddecay', 'adagrad']. Use adamasync by default.
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
      - `--learning_rate`: Learning rate for deep network. Default to 0.001.
      - `--tf`: Use TF 1.15.5 API and disable DeepRec features.

## Dataset
Train & eval dataset using [***Ali ELM dataset***](https://tianchi.aliyun.com/dataset/dataDetail?dataId=131047).

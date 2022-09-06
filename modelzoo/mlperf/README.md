# MLPerf

This model is DLRM_DCN, for MLPerf 2022.

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
        cd /root/modelzoo/DLRM
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
      - `--optimizer`: Choose the optimizer for deep model from ['adam', 'adamasync', 'adagraddecay', 'adagrad', 'gradientdescent']. Use adamasync by default.
      - `--smartstaged`: Whether to enable smart staged feature of DeepRec, Default to True.
      - `--micro_batch`: Set num for Auto Mirco Batch. Default 0 to close.(Not really enabled)
      - `--ev`: Whether to enable DeepRec EmbeddingVariable. Default to False.
      - `--adaptive_emb`: Whether to enable Adaptive Embedding. Default to False.
      - `--ev_elimination`: Set Feature Elimination of EmbeddingVariable Feature. Options [None, 'l2', 'gstep'], default to None.
      - `--ev_filter`: Set Feature Filter of EmbeddingVariable Feature. Options [None, 'counter', 'cbf'], default to None.
      - `--dynamic_ev`: Whether to enable Dynamic-dimension Embedding Variable. Default to False.(Not really enabled)
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
      - `--learning_rate`: Learning rate for network. Default to 0.1.
      - `--interaction_op`: Choose interaction op before top MLP layer('dot', 'cat'). Default to cat.
      - `--inter`: Set inter op parallelism threads. Default to 0.
      - `--intra`: Set intra op parallelism threads. Default to 0.
      - `--input_layer_partitioner`: Slice size of input layer partitioner(units MB).
      - `--dense_layer_partitioner`: Slice size of dense layer partitioner(units kB).
      - `--tf`: Use TF 1.15.5 API and disable DeepRec features.

### Processing
- Interger columns **I[1-13]** is processed with `tf.feature_column.numeric_column()` function, and the data is normalized.  
    In order to save time, the data required for normalization has been calculated in advance.
- Categorical columns **C[1-26]** is processed with `tf.feature_column.embedding_column()` function after using `tf.feature_column.categorical_column_with_hash_bucket()` function.

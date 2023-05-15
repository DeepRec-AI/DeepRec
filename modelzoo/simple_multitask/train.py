import argparse
import collections
import json
import math
import os
import tensorflow as tf
from tensorflow.python.client import timeline as tf_timeline
from tensorflow.python.ops import partitioned_variables
import time

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print(f'Using TensorFlow version {tf.__version__}')

LABEL_COLUMNS = ['clk', 'buy']
HASH_INPUTS = [
        'pid',
        'adgroup_id',
        'cate_id',
        'campaign_id',
        'customer',
        'brand',
        'user_id',
        'cms_segid',
        'cms_group_id',
        'final_gender_code',
        'age_level',
        'pvalue_level',
        'shopping_level',
        'occupation',
        'new_user_class_level',
        'tag_category_list',
        'tag_brand_list',
        'price'
        ]

HASH_BUCKET_SIZES = {
        'pid': 10,
        'adgroup_id': 100000,
        'cate_id': 10000,
        'campaign_id': 100000,
        'customer': 100000,
        'brand': 100000,
        'user_id': 100000,
        'cms_segid': 100,
        'cms_group_id': 100,
        'final_gender_code': 10,
        'age_level': 10,
        'pvalue_level': 10,
        'shopping_level': 10,
        'occupation': 10,
        'new_user_class_level': 10,
        'tag_category_list': 100000,
        'tag_brand_list': 100000,
        'price': 50
        }
defaults = [[0]] * len(LABEL_COLUMNS) + [[' ']] * len(HASH_INPUTS)
headers = LABEL_COLUMNS + HASH_INPUTS

class SimpleMultiTask():
    def __init__(self,
                 input,
                 feature_columns,
                 mlp=[256, 196, 128, 64],
                 batch_size=None,
                 optimizer_type='adam',
                 bf16=False,
                 stock_tf=None,
                 adaptive_emb=False,
                 learning_rate=0.1,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not input:
            raise ValueError('Dataset is not defined.')
        if not feature_columns:
            raise ValueError('Feature columns are not defined.')

        self._feature_columns = feature_columns
        self._mlp = mlp
        self._batch_size = batch_size
        self._optimizer_type = optimizer_type
        self._learning_rate = learning_rate
        self._tf = stock_tf
        self._adaptive_emb = adaptive_emb
        self._bf16 = False if self._tf else bf16

        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner

        self.feature = input[1]
        self._labels = input[0]
        self.label = tf.stack([self._labels['clk'], self._labels['buy']], axis=1)

        self._create_model()

        with tf.name_scope('head'):
            self._create_loss()
            self._create_optimizer()
            self._create_metrics()

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    # compute loss
    def _create_loss(self):
        bce_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.probability = tf.squeeze(self.probability)
        self.loss = tf.math.reduce_mean(bce_loss_func(self.label, self.probability))
        tf.summary.scalar('loss', self.loss)

    # define optimizer and generate train_op
    def _create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        if (self._tf and self._optimizer_type == 'adamasync') or self._optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._learning_rate)
        elif not self._tf and self._optimizer_type == 'adamasync':
            optimizer = tf.train.AdamAsyncOptimizer(learning_rate=self._learning_rate)
        elif self._optimizer_type == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
        elif self._optimizer_type == 'adagraddecay':
            optimizer = tf.train.AdagradDecayOptimizer(learning_rate=self._learning_rate,
                                                       global_step=self.global_step)
        elif self._optimizer_type == 'gradientdescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
        else:
            raise ValueError('Optimizer type error.')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss,
                                               global_step=self.global_step)

    # compute acc & auc
    def _create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self.label,
                                                    predictions=self.output)
        self.auc, self.auc_op = tf.metrics.auc(labels=self.label,
                                               predictions=self.probability,
                                               num_thresholds=1000)
        tf.summary.scalar('eval_acc', self.acc)
        tf.summary.scalar('eval_auc', self.auc)

    def _create_dense_layer(self, input, num_hidden_units, activation, layer_name):
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE) as mlp_layer_scope:
            dense_layer = tf.layers.dense(input,
                                       units=num_hidden_units,
                                       activation=activation,
                                       name=mlp_layer_scope)
            self._add_layer_summary(dense_layer, mlp_layer_scope.name)
        return dense_layer

    def _make_scope(self, name, bf16):
        if(bf16):
            return tf.variable_scope(name, reuse=tf.AUTO_REUSE).keep_weights(dtype=tf.float32)
        else:
            return tf.variable_scope(name, reuse=tf.AUTO_REUSE)

    # create model
    def _create_model(self):
        with tf.variable_scope('input_layer',
                                partitioner=self._input_layer_partitioner,
                                reuse=tf.AUTO_REUSE):
            if not self._tf and self._adaptive_emb:
                '''Adaptive Embedding Feature Part 1 of 2'''
                adaptive_mask_tensors = {}
                for i in range(len(HASH_INPUTS)):
                    adaptive_mask_tensors[HASH_INPUTS[i]] = tf.ones([self._batch_size],
                                                                    tf.int32)
                self._input_layer = tf.feature_column.input_layer(
                        self.feature,
                        self._feature_columns,
                        adaptive_mask_tensors=adaptive_mask_tensors)
            else:
                self._input_layer = tf.feature_column.input_layer(self.feature, self._feature_columns)

            if self._bf16:
                self._input_layer = tf.cast(self._input_layer, dtype=tf.bfloat16)

        with tf.variable_scope('SimpleMultiTask',
                               reuse=tf.AUTO_REUSE):
            self.clk_model, self.clk_logits = self._build_clk_model()
            self.buy_model, self.buy_logits = self._build_buy_model()
            self._logits = tf.squeeze(tf.stack([self.clk_model, self.buy_model], axis=1), [-1])
            self.probability = self._logits
            self.output = tf.round(self.probability)

    def _build_clk_model(self):
        with self._make_scope('clk_model', self._bf16):
            if self._bf16:
                self._input_layer = tf.cast(self._input_layer, dtype=tf.bfloat16)
            d_clk = self._input_layer
            for layer_id, num_hidden_units in enumerate(self._mlp):
                d_clk = self._create_dense_layer(d_clk, num_hidden_units, tf.nn.relu, f'd{layer_id}_clk')

            d_clk = self._create_dense_layer(d_clk, 1, None, 'output_clk')
            if self._bf16:
                d_clk = tf.cast(d_clk, tf.float32)

        Y_clk = tf.squeeze(d_clk)

        return tf.math.sigmoid(d_clk), Y_clk

    def _build_buy_model(self):
        with self._make_scope('buy_model', self._bf16):
            if self._bf16:
                self._input_layer = tf.cast(self._input_layer, dtype=tf.bfloat16)
            d_buy = self._input_layer
            for layer_id, num_hidden_units in enumerate(self._mlp):
                d_buy = self._create_dense_layer(d_buy, num_hidden_units, tf.nn.relu, f'd{layer_id}_buy')

            d_buy = self._create_dense_layer(d_buy, 1, None, 'output_buy')
            if self._bf16:
                d_buy = tf.cast(d_buy, tf.float32)

        Y_buy = tf.squeeze(d_buy)

        return tf.math.sigmoid(d_buy), Y_buy

# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs, seed, stock_tf, workqueue=None):
    def parse_csv(value):
        l = list(zip(headers, tf.io.decode_csv(value, defaults)))
        # This is because Dataset.map() have strange requirement of using collections.OrderedDict
        # otherwise throws type exception.
        return collections.OrderedDict(l[:2]), collections.OrderedDict(l[2:])

    def parse_parquet(value):
        tf.logging.info('Parsing {}'.format(filename))
        labels = {}
        for i in range(0, len(LABEL_COLUMNS)):
            labels[LABEL_COLUMNS[i]] = value.pop(LABEL_COLUMNS[i])
        features = value
        return labels, features

    '''Work Queue Feature'''
    if not stock_tf and workqueue:
        from tensorflow.python.ops.work_queue import WorkQueue
        work_queue = WorkQueue([filename], num_epochs=num_epochs)
        files = work_queue.input_dataset()
    else:
        files = filename
    if args.parquet_dataset and not args.tf:
        from tensorflow.python.data.experimental.ops import parquet_dataset_ops
        dataset = parquet_dataset_ops.ParquetDataset(files, batch_size=batch_size)
        if args.parquet_dataset_shuffle:
            dataset = dataset.shuffle(buffer_size=40000,
                                      seed=seed)  # fix seed for reproducing
        if not workqueue:
            dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(parse_parquet, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = tf.data.TextLineDataset(files)
        dataset = dataset.shuffle(buffer_size=400000,
                                  seed=seed)  # set seed for reproducing
        if not workqueue:
            dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(parse_csv, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(2)
    return dataset

# generate feature columns
def build_feature_columns(stock_tf,
                          emb_fusion,
                          emb_variable=None,
                          emb_var_elimination=None,
                          emb_var_filter=None,
                          adaptive_emb=None,
                          dynamic_emb_var=None):
    feature_columns = []
    if args.group_embedding and not args.tf:
        with tf.feature_column.group_embedding_column_scope(name="categorical"):
            for i in range(len(HASH_INPUTS)):
                column = tf.feature_column.categorical_column_with_hash_bucket(
                    HASH_INPUTS[i],
                    hash_bucket_size=HASH_BUCKET_SIZES[HASH_INPUTS[i]],
                    dtype=tf.string)

                if not stock_tf and (emb_variable or adaptive_emb or dynamic_emb_var):
                    '''Feature Elimination of EmbeddingVariable Feature'''
                    if emb_var_elimination == 'gstep':
                        # Feature elimination based on global steps
                        evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
                    elif emb_var_elimination == 'l2':
                        # Feature elimination based on l2 weight
                        evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)
                    else:
                        evict_opt = None

                    '''Feature Filter of EmbeddingVariable Feature'''
                    if emb_var_filter == 'cbf':
                        # CBF-based feature filter
                        filter_option = tf.CBFFilter(filter_freq=3,
                                                    max_element_size=2**30,
                                                    false_positive_probability=0.01,
                                                    counter_type=tf.int64)
                    elif emb_var_filter == 'counter':
                        # Counter-based feature filter
                        filter_option = tf.CounterFilter(filter_freq=3)
                    else:
                        filter_option = None

                    ev_opt = tf.EmbeddingVariableOption(evict_option=evict_opt, filter_option=filter_option)

                    if emb_variable:
                        '''Embedding Variable Feature'''
                        column = tf.feature_column.categorical_column_with_embedding(HASH_INPUTS[i],
                                                                                    dtype=tf.string,
                                                                                    ev_option=ev_opt)
                    elif adaptive_emb:
                        '''                 Adaptive Embedding Feature Part 2 of 2
                        Except the following code, a dict, 'adaptive_mask_tensors', is needede as the input of
                        'tf.feature_column.input_layer(adaptive_mask_tensors=adaptive_mask_tensors)'.
                        For column 'COL_NAME',the value of adaptive_mask_tensors['$COL_NAME'] is an int32
                        tensor with shape [batch_size].
                        '''
                        column = tf.feature_column.categorical_column_with_adaptive_embedding(
                            HASH_INPUTS[i],
                            hash_bucket_size=HASH_BUCKET_SIZES[HASH_INPUTS[i]],
                            dtype=tf.string,
                            ev_option=ev_opt)
                    elif dynamic_emb_var:
                        '''Dynamic-dimension Embedding Variable'''
                        raise ValueError('Dynamic-dimension Embedding Variable is not enabled in the model')

                if not stock_tf and emb_fusion:
                    '''Embedding Fusion Feature'''
                    embedding_column = tf.feature_column.embedding_column(column,
                                                                        dimension=16,
                                                                        combiner='mean',
                                                                        do_fusion=emb_fusion)
                else:
                    embedding_column = tf.feature_column.embedding_column(column,
                                                            dimension=16,
                                                            combiner='mean')


                feature_columns.append(embedding_column)
    else:
        for i in range(len(HASH_INPUTS)):
            column = tf.feature_column.categorical_column_with_hash_bucket(
                HASH_INPUTS[i],
                hash_bucket_size=HASH_BUCKET_SIZES[HASH_INPUTS[i]],
                dtype=tf.string)

            if not stock_tf and (emb_variable or adaptive_emb or dynamic_emb_var):
                '''Feature Elimination of EmbeddingVariable Feature'''
                if emb_var_elimination == 'gstep':
                    # Feature elimination based on global steps
                    evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
                elif emb_var_elimination == 'l2':
                    # Feature elimination based on l2 weight
                    evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)
                else:
                    evict_opt = None

                '''Feature Filter of EmbeddingVariable Feature'''
                if emb_var_filter == 'cbf':
                    # CBF-based feature filter
                    filter_option = tf.CBFFilter(filter_freq=3,
                                                max_element_size=2**30,
                                                false_positive_probability=0.01,
                                                counter_type=tf.int64)
                elif emb_var_filter == 'counter':
                    # Counter-based feature filter
                    filter_option = tf.CounterFilter(filter_freq=3)
                else:
                    filter_option = None

                ev_opt = tf.EmbeddingVariableOption(evict_option=evict_opt, filter_option=filter_option)

                if emb_variable:
                    '''Embedding Variable Feature'''
                    column = tf.feature_column.categorical_column_with_embedding(HASH_INPUTS[i],
                                                                                dtype=tf.string,
                                                                                ev_option=ev_opt)
                elif adaptive_emb:
                    '''                 Adaptive Embedding Feature Part 2 of 2
                    Except the following code, a dict, 'adaptive_mask_tensors', is needede as the input of
                    'tf.feature_column.input_layer(adaptive_mask_tensors=adaptive_mask_tensors)'.
                    For column 'COL_NAME',the value of adaptive_mask_tensors['$COL_NAME'] is an int32
                    tensor with shape [batch_size].
                    '''
                    column = tf.feature_column.categorical_column_with_adaptive_embedding(
                        HASH_INPUTS[i],
                        hash_bucket_size=HASH_BUCKET_SIZES[HASH_INPUTS[i]],
                        dtype=tf.string,
                        ev_option=ev_opt)
                elif dynamic_emb_var:
                    '''Dynamic-dimension Embedding Variable'''
                    raise ValueError('Dynamic-dimension Embedding Variable is not enabled in the model')

            if not stock_tf and emb_fusion:
                '''Embedding Fusion Feature'''
                embedding_column = tf.feature_column.embedding_column(column,
                                                                    dimension=16,
                                                                    combiner='mean',
                                                                    do_fusion=emb_fusion)
            else:
                embedding_column = tf.feature_column.embedding_column(column,
                                                        dimension=16,
                                                        combiner='mean')


            feature_columns.append(embedding_column)

    return feature_columns

def train(sess_config,
          input_hooks,
          model,
          train_init_op,
          train_steps,
          keep_checkpoint_max,
          checkpoint_dir,
          save_steps=None,
          timeline_steps=None,
          no_eval=None,
          tf_config=None,
          server=None,
          stock_tf=None,
          incremental_ckpt=None):
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), train_init_op),
        saver=tf.train.Saver(max_to_keep=keep_checkpoint_max))

    stop_hook = tf.train.StopAtStepHook(last_step=train_steps)
    log_hook = tf.train.LoggingTensorHook(
        {'steps': model.global_step,
         'loss': model.loss
        }, every_n_iter=100)
    hooks.append(stop_hook)
    hooks.append(log_hook)

    if timeline_steps and timeline_steps > 0:
        hooks.append(tf.train.ProfilerHook(save_steps=timeline_steps,
                                           output_dir=checkpoint_dir))
    save_ckp_steps = save_steps if save_steps or no_eval else train_steps

    '''
                            Incremental_Checkpoint
    To enable Incremental_Checkpoint please `save_incremental_checkpoint_secs`
    in 'tf.train.MonitoredTrainingSession' checkpoint_dir must be defined. By default
    save_incremental_checkpoint_secs is None. Incremental_save checkpoint time
    in seconds can be set to use incremental checkpoint function, like
    `tf.train.MonitoredTrainingSession(save_incremental_checkpoint_secs=args.incremental_ckpt)`
    '''
    if not stock_tf and incremental_ckpt:
        raise ValueError('Incremental_Checkpoint has not been enabled yet. ' \
                         'Please see comments in the code.')
    else:
        with tf.train.MonitoredTrainingSession(
                master=server.target if server else '',
                is_chief=tf_config['is_chief'] if tf_config else True,
                hooks=hooks,
                scaffold=scaffold,
                checkpoint_dir=checkpoint_dir,
                save_checkpoint_steps=save_ckp_steps,
                summary_dir=checkpoint_dir,
                save_summaries_steps=save_steps,
                config=sess_config) as sess:
            while not sess.should_stop():
                sess.run([model.loss, model.train_op])
    print('Training completed.')


def eval(sess_config, input_hooks, model, test_init_op, test_steps, output_dir, checkpoint_dir):
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), test_init_op))
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
    writer = tf.summary.FileWriter(os.path.join(output_dir, 'eval'))
    merged = tf.summary.merge_all()

    with tf.train.MonitoredSession(session_creator=session_creator,
                                   hooks=hooks) as sess:
        for _in in range(1, test_steps + 1):
            if (_in != test_steps):
                sess.run([model.acc_op, model.auc_op])
                if (_in % 1000 == 0):
                    print(f'Evaluation complete:[{_in}/{test_steps}]')
            else:
                eval_acc, eval_auc, events = sess.run([model.acc_op, model.auc_op, merged])
                writer.add_summary(events, _in)
                print(f'Evaluation complete:[{_in}/{test_steps}]')
                print(f'ACC = {eval_acc}\nAUC = {eval_auc}')

def main(stock_tf, tf_config=None, server=None):
    # check dataset and count data set size
    print('Checking dataset...')
    train_file = args.data_location + '/taobao_train_data'
    test_file = args.data_location + '/taobao_test_data'
    if args.parquet_dataset and not args.tf:
        train_file += '.parquet'
        test_file += '.parquet'

    if not os.path.exists(args.data_location):
        raise ValueError(f'[ERROR] data location: {args.data_location} does not exist. '
                         'Please provide valid path')

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise ValueError('[ERROR] taobao_train_data or taobao_test_data does not exist '
                         'in the given data_location. Please provide valid path')

    no_of_training_examples = 0
    no_of_test_examples = 0
    if args.parquet_dataset and not args.tf:
        import pyarrow.parquet as pq
        no_of_training_examples = pq.read_table(train_file).num_rows
        no_of_test_examples = pq.read_table(test_file).num_rows
    else:
        no_of_training_examples = sum(1 for _ in open(train_file))
        no_of_test_examples = sum(1 for _ in open(test_file))

    # set batch size, eporch & steps
    batch_size = math.ceil(
                  args.batch_size / args.micro_batch
                  ) if args.micro_batch and not stock_tf else args.batch_size
    if args.steps == 0:
        no_epochs = 100
        train_steps = math.ceil(
            (float(no_epochs) * no_of_training_examples) / batch_size)
    else:
        no_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)

    print(f'Number of batch size: {batch_size}')

    print(f'Numbers of training dataset: {no_of_training_examples}')
    print(f'Number of epochs: {no_epochs}')
    print(f'Number of train steps: {train_steps}')

    print(f'Numbers of test dataset: {no_of_test_examples}')
    print(f'Numbers of test steps: {test_steps}')

    if not stock_tf:
        print('Optimizations')
        print(f'\tNumber of micro batches: {args.micro_batch}')
        print(f'\tEmbedding Fusion Feature: {args.emb_fusion}')
        print(f'\tSmartStage Feature: {args.smartstaged}')
        print(f'\tAuto Graph Fusion: {args.op_fusion}')
        print(f'\tEmbeddingVariable: {args.ev}')
        print(f'\tFeature Elimination of EmbeddingVariable Feature: {args.ev_elimination}')
        print(f'\tFeature Filter of EmbeddingVariable Feature: {args.ev_filter}')
        print(f'\tAdaptive Embedding: {args.adaptive_emb}')
        print(f'\tDynamic-dimension Embedding Variable: {args.dynamic_ev}')
        print(f'\tIncremental Checkpoint: {args.incremental_ckpt}')
        print(f'\tWorkQueue: {args.workqueue}')
    temp_optimizer = 'adam' if (stock_tf and args.optimizer == 'adamasync') else args.optimizer
    print(f'Used optimizer: {temp_optimizer}')

    # set fixed random seed
    SEED = args.seed
    tf.set_random_seed(SEED)

    # set directory path for checkpoint_dir
    keep_checkpoint_max = args.keep_checkpoint_max
    model_dir = os.path.join(args.output_dir, 'model_SimpleMultiTask_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else model_dir
    print(f'Saving model events to {checkpoint_dir}')

    if keep_checkpoint_max:
        print(f'Maximum number of saved checkpoints: {keep_checkpoint_max}')

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_epochs, SEED, stock_tf, args.workqueue)
    test_dataset = build_model_input(test_file, batch_size, 1, SEED, stock_tf, args.workqueue)

    dataset_output_types = tf.data.get_output_types(train_dataset)
    dataset_output_shapes = tf.data.get_output_shapes(test_dataset)
    iterator = tf.data.Iterator.from_structure(dataset_output_types,
                                               dataset_output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create feature column
    feature_columns = build_feature_columns(stock_tf,
                                            args.emb_fusion,
                                            args.ev,
                                            args.ev_elimination,
                                            args.ev_filter,
                                            args.adaptive_emb,
                                            args.dynamic_ev)

    # create variable partitioner for distributed training
    num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
    input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
                                max_partitions=num_ps_replicas,
                                min_slice_size=args.input_layer_partitioner <<
                                    20) if args.input_layer_partitioner else None
    dense_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
                                max_partitions=num_ps_replicas,
                                min_slice_size=args.dense_layer_partitioner <<
                                    10) if args.dense_layer_partitioner else None

    # Session config
    sess_config = tf.ConfigProto()
    if tf_config:
        sess_config.device_filters.append("/job:ps")
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    # Session hook
    hooks = []

    if args.smartstaged and not stock_tf:
        '''SmartStage Feature'''
        next_element = tf.staged(next_element, num_threads=4, capacity=40)
        sess_config.graph_options.optimizer_options.do_smart_stage = True
        hooks.append(tf.make_prefetch_hook())
    if args.op_fusion and not stock_tf:
        '''Auto Graph Fusion'''
        sess_config.graph_options.optimizer_options.do_op_fusion = True
    if args.micro_batch and not stock_tf:
        '''Auto Micro Batch'''
        sess_config.graph_options.optimizer_options.micro_batch_num = args.micro_batch

    # create model
    model = SimpleMultiTask(next_element,
                 feature_columns,
                 batch_size=batch_size,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=stock_tf,
                 adaptive_emb=args.adaptive_emb,
                 learning_rate=args.learning_rate,
                 input_layer_partitioner=input_layer_partitioner,
                 dense_layer_partitioner=dense_layer_partitioner)

    # Run model training and evaluation
    train(sess_config, hooks, model, train_init_op, train_steps,
          keep_checkpoint_max, checkpoint_dir, args.save_steps,
          args.timeline, args.no_eval, tf_config, server,
          stock_tf, args.incremental_ckpt)
    if not (args.no_eval or tf_config):
        eval(sess_config, hooks, model, test_init_op, test_steps,
             model_dir, checkpoint_dir)

def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='set random seed', type=int, default=2020)
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train',
                        type=int,
                        default=2048)
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        default='./result')
    parser.add_argument('--checkpoint_dir',
                        help='Full path to checkpoints output directory')
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.1)
    parser.add_argument('--l2_regularization',
                        help='L2 regularization for the model',
                        type=float)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline',
                        type=int)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset',
                        action='store_true')
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc', 'grpc++', 'star_server'],
                        default='grpc')
    parser.add_argument('--inter',
                        help='set inter op parallelism threads',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set intra op parallelism threads',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner',
                        help='slice size of input layer partitioner. units MB',
                        type=int,
                        default=0)
    parser.add_argument('--dense_layer_partitioner',
                        help='slice size of dense layer partitioner. units KB',
                        type=int,
                        default=0)
    parser.add_argument('--tf',
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    '''Enabled By Default DeepRec optimizations'''
    parser.add_argument('--smartstaged', \
                        help='Whether to enable SmartStage feature of DeepRec',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--emb_fusion', \
                        help='Whether to enable embedding fusion',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--op_fusion', \
                        help='Whether to enable Auto graph fusion feature',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--micro_batch',
                        help='Set number for Auto Micro Batch',
                        type=int,
                        default=0)
    parser.add_argument('--optimizer', type=str,
                        choices=['adam', 'adamasync', 'adagraddecay',
                                 'adagrad', 'gradientdescent'],
                        default='adamasync')
    '''Optional DeepRec optimizations'''
    parser.add_argument('--ev', \
                        help='Whether to enable DeepRec EmbeddingVariable',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev_elimination', \
                        help='Feature Elimination of EmbeddingVariable Feature',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter', \
                        help='Feature Filter of EmbeddingVariable Feature',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--adaptive_emb', \
                        help='Whether to enable Adaptive Embedding',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--dynamic_ev', \
                        help='Whether to enable Dynamic-dimension Embedding Variable',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--incremental_ckpt', \
                        help='Set time[in seconds] of save Incremental Checkpoint',
                        type=int,
                        default=None)
    parser.add_argument('--workqueue', \
                        help='Whether to enable WorkQueue',
                        type=boolean_string,
                        default=False)
    parser.add_argument("--parquet_dataset", \
                        help='Whether to enable Parquet DataSet. Defualt to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument("--parquet_dataset_shuffle", \
                        help='Whether to enable shuffle operation for Parquet Dataset. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument("--group_embedding", \
                        help='Whether to enable Group Embedding. Defualt to None.',
                        type=str,
                        choices=[None, 'localized', 'collective'],
                        default=None)

    return parser

# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    print(f'Running distributed training with TF_CONFIG: {TF_CONFIG}')

    tf_config = json.loads(TF_CONFIG)
    cluster_config = tf_config.get('cluster')
    ps_hosts = []
    worker_hosts = []
    chief_hosts = []
    for key, value in cluster_config.items():
        if 'ps' == key:
            ps_hosts = value
        elif 'worker' == key:
            worker_hosts = value
        elif 'chief' == key:
            chief_hosts = value
    if chief_hosts:
        worker_hosts = chief_hosts + worker_hosts

    if not ps_hosts or not worker_hosts:
        raise ValueError(f'[TF_CONFIG ERROR] Incorrect ps_hosts or incorrect worker_hosts')

    task_config = tf_config.get('task')
    task_type = task_config.get('type')
    task_index = task_config.get('index') + (1 if task_type == 'worker'
                                             and chief_hosts else 0)

    if task_type == 'chief':
        task_type = 'worker'

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    if args.protocol in ['grpc++', 'star_server']:
        raise ValueError('grpc++ and star_server protocols have not been enabled in the model yet')

    server = tf.distribute.Server(cluster,
                                  job_name=task_type,
                                  task_index=task_index,
                                  protocol=args.protocol)
    if task_type == 'ps':
        server.join()
    elif task_type == 'worker':
        tf_config = {'ps_hosts': ps_hosts,
                     'worker_hosts': worker_hosts,
                     'type': task_type,
                     'index': task_index,
                     'is_chief': is_chief}

        tf_device = tf.device(tf.train.replica_device_setter(
                              worker_device=f'/job:worker/task:{task_index}',
                              cluster=cluster))
        return tf_config, server, tf_device
    else:
        raise ValueError(f'[TF_CONFIG ERROR] Task type or index error.')

# Some DeepRec's features are enabled by ENV.
# This func is used to set ENV and enable these features.
# A triple quotes comment is used to introduce these features and play an emphasizing role.
def set_env_for_DeepRec():
    '''
    Set some ENV for these DeepRec's features enabled by ENV.
    More Detail information is shown in https://deeprec.readthedocs.io/zh/latest/index.html.
    START_STATISTIC_STEP & STOP_STATISTIC_STEP: On CPU platform, DeepRec supports memory optimization
        in both stand-alone and distributed trainging. It's default to open, and the
        default start and stop steps of collection is 1000 and 1100. Reduce the initial
        cold start time by the following settings.
    MALLOC_CONF: On CPU platform, DeepRec can use memory optimization with the jemalloc library.
        Please preload libjemalloc.so by `LD_PRELOAD=./libjemalloc.so.2 python ...`
    DNNL_MAX_CPU_ISA: Specify the highest instruction set used by oneDNN (when the version is less than 2.5.0),
        it will be set to AVX512_CORE_AMX to enable Intel CPU's feature.
    '''
    os.environ['START_STATISTIC_STEP'] = '100'
    os.environ['STOP_STATISTIC_STEP'] = '110'
    os.environ['MALLOC_CONF'] = \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'
    if args.group_embedding == "collective":
        tf.config.experimental.enable_distributed_strategy(strategy="collective")

def check_stock_tf():
    import pkg_resources
    detailed_version = pkg_resources.get_distribution('Tensorflow').version
    return not ('deeprec' in detailed_version)

def check_DeepRec_features():
    return args.smartstaged or args.emb_fusion or args.op_fusion or args.micro_batch or args.bf16 or \
           args.ev or args.adaptive_emb or args.dynamic_ev or (args.optimizer == 'adamasync') or \
           args.incremental_ckpt  or args.workqueue and args.group_embedding

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    stock_tf = args.tf
    if not stock_tf and check_stock_tf() and check_DeepRec_features():
        raise ValueError('Stock Tensorflow does not support DeepRec features. '
                         'For Stock Tensorflow run the script with `--tf` argument.')

    if not stock_tf:
        set_env_for_DeepRec()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        print('Running stand-alone mode training')
        main(stock_tf)
    else:
        tf_config, server, tf_device = generate_cluster_info(TF_CONFIG)
        with tf_device:
            main(stock_tf, tf_config, server)

import time
import argparse
import numbers
import tensorflow as tf
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
import json

from tensorflow.python.ops import partitioned_variables

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

'''
INPUT CONFIG SPECIFICS
'''
TRAIN_DATA_NAME="taobao_train_data"
TEST_DATA_NAME="taobao_test_data"

LABEL_COLUMNS = ["clk", "buy"]
HASH_INPUTS = [
    "pid",
    "adgroup_id",
    "cate_id",
    "campaign_id",
    "customer",
    "brand",
    "user_id",
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
    "tag_category_list",
    "tag_brand_list"
    ]
IDENTITY_INPUTS = ["price"]
ALL_FEATURE_COLUMNS = HASH_INPUTS + IDENTITY_INPUTS
ALL_INPUT = LABEL_COLUMNS + HASH_INPUTS + IDENTITY_INPUTS
NOT_USED_CATEGORY = ["final_gender_code"]

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
    'tag_brand_list': 100000
}

NUM_BUCKETS = {
    'price': 50
}

'''
END OF INPUT CONFIG SPECIFICS
'''
'''
MODEL CONFIG SPECIFICS
'''
DNN_ACTIVATION = tf.nn.relu

L2_REGULARIZATION = 1e-06
EMBEDDING_REGULARIZATION = 5e-05

EXPERTS_COUNT = 4
EXPERT_HIDDEN_UNITS = [256, 192, 128, 64]
EMBEDDING_DIM = 16

#Tower tuple structure (tower name, label name, hidden units)
TOWERS = [
    ("ctr", "clk", [256, 192, 128, 64]),
    ("cvr", "buy", [256, 192, 128, 64])
]
'''
MODEL CONFIG SPECIFICS
'''
def l2_regularizer(scale, scope=None):
  if isinstance(scale, numbers.Integral):
    raise ValueError(f'Scale cannot be an integer: {scale}')
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError(f'Setting a scale less than 0 on a regularizer: {scale}.')
    if scale == 0.:
      return lambda _: None

  def l2(weights):
    with tf.name_scope(scope, 'l2_regularizer', [weights]) as name:
      my_scale = tf.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
      return tf.math.multiply(my_scale, tf.nn.l2_loss(weights), name=name)

  return l2

class MMOE():
    def __init__(self,
                 input,
                 feature_column,
                 num_experts,
                 expert_hidden_units,
                 towers,
                 optimizer_type='adam',
                 learning_rate=0.1,
                 bf16=False,
                 stock_tf=None,
                 adaptive_emb=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not input:
            raise ValueError("Dataset is not defined.")
        self._feature = input[0]
        self._label = input[1]

        self._feature_column = feature_column
        self._num_experts = num_experts
        self._expert_hidden_units = expert_hidden_units
        self._towers = towers

        self._learning_rate = learning_rate
        self.tf = stock_tf
        self._bf16 = False if self.tf else bf16

        self.is_training = True
        self._adaptive_emb = adaptive_emb
        self._optimizer_type = optimizer_type
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner

        self.model = self._create_model()
        with tf.name_scope('head'):
            self._create_loss()
            self._create_optimizer()
            self._create_metrics()
    
    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                        tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)
    
    def _make_scope(self, name, bf16, part):
        if(bf16):
            return tf.variable_scope(name, partitioner=part, reuse=tf.AUTO_REUSE).keep_weights(dtype=tf.float32)
        else:
            return tf.variable_scope(name, partitioner=part, reuse=tf.AUTO_REUSE)

    # create model
    def _create_model(self):
        TAG_COLUMN = ['tag_category_list', 'tag_brand_list']
        for key in TAG_COLUMN:
            self._feature[key] = tf.strings.split(self._feature[key], '|')

        key_dict = {}
        with self._make_scope('input_layer', self._bf16, self._input_layer_partitioner):
            print('Adaptive emb = ', self._adaptive_emb, 'TF = ', self.tf)
            if self._adaptive_emb and not self.tf:
                '''Adaptive Embedding Feature part 1 of 2'''
                print('Adaptive Embedding Feature part 1 of 2')
                adaptive_mask_tensors = {}
                for col in HASH_INPUTS:
                    adaptive_mask_tensors[col] = tf.ones([args.batch_size],
                                                          tf.int32)
                input_emb = tf.feature_column.input_layer(
                    self._feature,
                    self._feature_column,
                    adaptive_mask_tensors=adaptive_mask_tensors,
                    cols_to_output_tensors=key_dict)
            else:
                input_emb = tf.feature_column.input_layer(
                    self._feature,
                    self._feature_column,
                    cols_to_output_tensors=key_dict)
        
        with self._make_scope('MMOE', self._bf16, self._dense_layer_partitioner):
            if self._bf16:
                input_emb = tf.cast(input_emb, dtype=tf.bfloat16)
            experts = []
            for i in range(1, self._num_experts + 1):
                with tf.variable_scope(f'expert_{i}'):
                    expert_features = input_emb

                    for layer_id, num_hidden_units in enumerate(self._expert_hidden_units):
                        with tf.variable_scope(f'expert_{i}_layer_{layer_id}', reuse=tf.AUTO_REUSE) as expert_layer_scope:
                            expert_features = tf.layers.dense(expert_features,
                                                            units=num_hidden_units,
                                                            activation=None,
                                                            name=f'{expert_layer_scope.name}/dense')
                            expert_features = DNN_ACTIVATION(expert_features, 
                                                            name=f'{expert_layer_scope.name}/act')               
                            self._add_layer_summary(expert_features, expert_layer_scope.name)
                    experts.append(expert_features)
            experts_features = tf.stack(experts, axis=1)

            towers=[]
            for tower in self._towers:
                tower_name = tower[0]
                hidden_units = tower[2]
                with tf.variable_scope(f'{tower_name}_gate', reuse=tf.AUTO_REUSE) as gate_scope:

                    gate = tf.layers.dense(input_emb,
                                        units=self._num_experts,
                                        name=f'{tower_name}_gate')
                    gate = tf.nn.softmax(gate, axis=1)
                    gate = tf.expand_dims(gate, -1)

                with tf.variable_scope(tower_name):
                    tower_input = expert_features

                    tower_input = tf.multiply(experts_features, gate)
                    tower_input = tf.reduce_sum(tower_input, axis=1)

                    tower_features = tower_input
                    for layer_id, num_hidden_units in enumerate(hidden_units):
                        with tf.variable_scope(f'{tower_name}_layer_{layer_id}', reuse=tf.AUTO_REUSE) as tower_layer_scope:
                            tower_features = tf.layers.dense(tower_features,
                                                            units=num_hidden_units,
                                                            name=f'{tower_layer_scope.name}/dense')
                            tower_features = DNN_ACTIVATION(tower_features,
                                                            name=f'{tower_layer_scope.name}/act')
                            self._add_layer_summary(tower_features, tower_layer_scope.name)
                    final_tower_predict = tf.layers.dense(inputs=tower_features,
                                                        units=1,
                                                        activation=None,
                                                        name=f'{tower_name}_output')
                    self._add_layer_summary(final_tower_predict, f'{tower_name}_output')
                    if self._bf16:
                        final_tower_predict = tf.cast(final_tower_predict, dtype=tf.float32)
                    towers.append(final_tower_predict)
            tower_stack = tf.stack(towers, axis=1)
            self._logits = tf.squeeze(tower_stack, [2])
            self.probability = tf.math.sigmoid(self._logits)
            self.output = tf.round(self.probability)
    
    # compute loss
    def _create_loss(self):
        self._logits = tf.squeeze(self._logits)
        self.loss = tf.losses.sigmoid_cross_entropy(
            self._label,
            self._logits,
            scope='loss',
            reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        tf.summary.scalar('loss', self.loss)
    
    # define optimizer and generate train_op
    def _create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        print('self.tf = ', self.tf, ' self._optimizer_type = ', self._optimizer_type)
        if self.tf or self._optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._learning_rate,
                initial_accumulator_value=0.1,
                use_locking=False)
        elif self._optimizer_type == 'adamasync':
            optimizer = tf.train.AdamAsyncOptimizer(
                learning_rate=self._learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagraddecay':
            optimizer = tf.train.AdagradDecayOptimizer(
                learning_rate=self._learning_rate,
                global_step=self.global_step)
        else:
            raise ValueError("Optimizer type error.")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss,
                                               global_step=self.global_step)
    
    # compute acc & auc
    def _create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self._label,
                                                    predictions=self.output)
        self.auc, self.auc_op = tf.metrics.auc(labels=self._label,
                                               predictions=self.probability,
                                               num_thresholds=1000)
        tf.summary.scalar('eval_acc', self.acc)
        tf.summary.scalar('eval_auc', self.auc)


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        HASH_defaults = [[" "] for i in range(0, len(HASH_INPUTS))]
        label_defaults = [[0] for i in range (0, len(LABEL_COLUMNS))]
        IDENTITY_defaults = [[0] for i in range(0, len(IDENTITY_INPUTS))]
        column_headers = LABEL_COLUMNS + HASH_INPUTS + IDENTITY_INPUTS
        record_defaults = label_defaults + HASH_defaults + IDENTITY_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = []
        for i in range(0, len(LABEL_COLUMNS)):
            labels.append(all_columns.pop(LABEL_COLUMNS[i]))
        label = tf.stack(labels, axis=1)
        features = all_columns
        return features, label
    
    '''Work Queue Feature'''
    if args.workqueue and not args.tf:
        from tensorflow.python.ops.work_queue import WorkQueue
        work_queue = WorkQueue([filename])
        files = work_queue.input_dataset()
    else:
        files = filename

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(files)
    dataset = dataset.shuffle(buffer_size=400000,
                              seed=args.seed)  # set seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(2)
    return dataset

# generate feature columns
def build_feature_cols():
    feature_cols = []
    for column_name in ALL_FEATURE_COLUMNS:
        if column_name in NOT_USED_CATEGORY:
            continue
        if column_name in HASH_INPUTS:
            print('Column name = ', column_name, ' hash bucket size = ', HASH_BUCKET_SIZES[column_name])
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name,
                hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                dtype=tf.string)
            
            if not args.tf:
                '''Feature Elimination of EmbeddingVariable Feature'''
                if args.ev_elimination == 'gstep':
                    # Feature elimination based on global steps
                    evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
                elif args.ev_elimination == 'l2':
                    # Feature elimination based on l2 weight
                    evict_opt = tf.L2WeightEvict(l2_weigt_threshold=1.0)
                else:
                    evict_opt = None
                '''Feature Filter of EmbeddingVariable Feature'''
                if args.ev_filter == 'cbf':
                    # CBF-based feature filter
                    filter_option = tf.CBFFilter(
                        filter_freq=3,
                        max_element_size=2**30,
                        false_positive_probability=0.01,
                        counter_type=tf.int64)
                elif args.ev_filter == 'counter':
                    # Counter-based feature filter
                    filter_option = tf.CounterFilter(filter_freq=3)
                else:
                    filter_option = None
                ev_opt = tf.EmbeddingVariableOption(
                    evict_option=evict_opt, filter_option=filter_option)

                if args.ev:
                    '''Embedding Variable Feature'''
                    categorical_column = tf.feature_column.categorical_column_with_embedding(
                        column_name, dtype=tf.string, ev_option=ev_opt)
                elif args.adaptive_emb:
                    '''                 Adaptive Embedding Feature Part 2 of 2
                    Expcet the follow code, a dict, 'adaptive_mask_tensors', is need as the input of 
                    'tf.feature_column.input_layer(adaptive_mask_tensors=adaptive_mask_tensors)'.
                    For column 'COL_NAME',the value of adaptive_mask_tensors['$COL_NAME'] is a int32
                    tensor with shape [batch_size].
                    '''

                    categorical_column = tf.feature_column.categorical_column_with_adaptive_embedding(
                        column_name,
                        hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                        dtype=tf.string,
                        ev_option=ev_opt)
                elif args.dynamic_ev:
                    '''Dynamic-dimension Embedding Variable'''
                    print("Dynamin-dimension Embedding Variable isn't really enabled in model.")
                    sys.exit()
            
            if args.tf or not args.emb_fusion:
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column, 
                    dimension=EMBEDDING_DIM, 
                    combiner='mean')
            else:
                '''Embedding Fusion Feature'''
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=EMBEDDING_DIM,
                    combiner='mean',
                    do_fusion=args.emb_fusion)

            feature_cols.append(embedding_column)

        elif column_name in IDENTITY_INPUTS:
            column = tf.feature_column.categorical_column_with_identity(column_name, 50)

            if args.tf or not args.emb_fusion:
                embedding_column = tf.feature_column.embedding_column(
                    column, 
                    dimension=EMBEDDING_DIM, 
                    combiner='mean')
            else:
                '''Embedding Fusion Feature'''
                embedding_column = tf.feature_column.embedding_column(
                    column,
                    dimension=EMBEDDING_DIM,
                    combiner='mean',
                    do_fusion=args.emb_fusion)

            feature_cols.append(embedding_column)
        else:
            raise ValueError('Unexpected column name occured')
    
    return feature_cols

def train(sess_config,
          input_hooks,
          model,
          data_init_op,
          steps,
          checkpoint_dir,
          tf_config=None,
          server=None):
    model.is_training = True
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), data_init_op),
        saver=tf.train.Saver(max_to_keep=args.keep_checkpoint_max))
    
    stop_hook = tf.train.StopAtStepHook(last_step=steps)
    log_hook = tf.train.LoggingTensorHook(
        {
            'steps': model.global_step,
            'loss': model.loss
        }, every_n_iter=100)
    hooks.append(stop_hook)
    hooks.append(log_hook)
    if args.timeline > 0:
        hooks.append(
            tf.train.ProfilerHook(save_steps=args.timeline,
                                  output_dir=checkpoint_dir))
    save_steps = args.save_steps if args.save_steps or args.no_eval else steps
    '''
                            Incremental_Checkpoint
    Please add `save_incremental_checkpoint_secs` in 'tf.train.MonitoredTrainingSession'
    it's default to None, Incremental_save checkpoint time in seconds can be set 
    to use incremental checkpoint function, like `tf.train.MonitoredTrainingSession(
        save_incremental_checkpoint_secs=args.incremental_ckpt)`
    '''
    if args.incremental_ckpt and not args.tf:
        print("Incremental_Checkpoint is not really enabled.")
        print("Please see the comments in the code.")
        sys.exit()
    
    with tf.train.MonitoredTrainingSession(
            master=server.target if server else '',
            is_chief=tf_config['is_chief'] if tf_config else True,
            hooks=hooks,
            scaffold=scaffold,
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=save_steps,
            summary_dir=checkpoint_dir,
            save_summaries_steps=args.save_steps,
            config=sess_config) as sess:
        while not sess.should_stop():
            sess.run([model.loss, model.train_op])
    print("Training completed.")

def eval(sess_config, input_hooks, model, data_init_op, steps, checkpoint_dir):
    model.is_training = False
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), data_init_op))
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
    writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval'))
    merged = tf.summary.merge_all()

    with tf.train.MonitoredSession(session_creator=session_creator,
                                   hooks=hooks) as sess:
        for _in in range(1, steps + 1):
            if (_in != steps):
                sess.run([model.acc_op, model.auc_op])
                if (_in % 1000 == 0):
                    print("Evaluation complete:[{}/{}]".format(_in, steps))
            else:
                eval_acc, eval_auc, events = sess.run(
                    [model.acc_op, model.auc_op, merged])
                writer.add_summary(events, _in)
                print("Evaluation complete:[{}/{}]".format(_in, steps))
                print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))

def main(tf_config=None, server=None):
        # check dataset and count data set size
    print("Checking dataset...")
    train_file = os.path.join(args.data_location, 'taobao_train_data')
    test_file = os.path.join(args.data_location, 'taobao_test_data')

    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print("Dataset does not exist in the given data_location.")
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Number of training dataset is {}".format(no_of_training_examples))
    print("Number of test dataset is {}".format(no_of_test_examples))

    # set batch size, epoch & steps
    batch_size = math.ceil(
        args.batch_size / args.micro_batch
    ) if args.micro_batch and not args.tf else args.batch_size

    if args.steps == 0:
        no_of_epochs = 1000
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)
    print("The training steps is {}".format(train_steps))
    print("The testing steps is {}".format(test_steps))

    # set fixed random seed
    tf.set_random_seed(args.seed)

    # ste directory path for checkpoint_dir
    model_dir = os.path.join(args.output_dir,
                             'model_MMOE_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to = " + checkpoint_dir)

    # create data pipeline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
    test_dataset = build_model_input(test_file, batch_size, 1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create future column
    feature_cols = build_feature_cols()

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

    # Session hooks
    hooks = []

    if args.smartstaged and not args.tf:
        '''Smart staged Feature'''
        next_element = tf.staged(next_element, num_threads=4, capacity=40)
        sess_config.graph_options.optimizer_options.do_smart_stage = True
        hooks.append(tf.make_prefetch_hook())
    if args.op_fusion and not args.tf:
        '''Auto Graph Fusion'''
        sess_config.graph_options.optimizer_options.do_op_fusion = True
    if args.micro_batch and not args.tf:
        '''Auto Mirco Batch'''
        sess_config.graph_options.optimizer_options.micro_batch_num = args.micro_batch

    # create model
    model = MMOE(input=next_element,
                 feature_column=feature_cols,
                 num_experts=EXPERTS_COUNT,
                 expert_hidden_units=EXPERT_HIDDEN_UNITS,
                 towers=TOWERS,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=args.tf,
                 adaptive_emb=args.adaptive_emb,
                 input_layer_partitioner=input_layer_partitioner,
                 dense_layer_partitioner=dense_layer_partitioner)

    # run model training and evalutaion
    train(sess_config, hooks, model, train_init_op, train_steps,
          checkpoint_dir, tf_config, server)
    if not (args.no_eval or tf_config):
        eval(sess_config, hooks, model, test_init_op, test_steps,
             checkpoint_dir)

def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output directory',
                        required=False)
    parser.add_argument('--model_dir',
                        help='Full path to test model directory',
                        required=False)
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.1)
    parser.add_argument('--l2_regularization',
                        help='L2 regularization for the model',
                        type=float,
                        default=L2_REGULARIZATION)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline',
                        type=int,
                        default=0)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--seed',
                        help='set random seed',
                        type=int,
                        default=2021)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
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
    parser.add_argument('--optimizer',
                        type=str, \
                        choices=['adam', 'adamasync', 'adagraddecay', 'adagrad'],
                        default='adamasync')
    parser.add_argument('--tf', \
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    parser.add_argument('--smartstaged', \
                        help='Whether to enable smart staged feature of DeepRec, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--emb_fusion', \
                        help='Whether to enable embedding fusion, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--ev', \
                        help='Whether to enable DeepRec EmbeddingVariable. Default False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev_elimination', \
                        help='Feature Elimination of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter', \
                        help='Feature Filter of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--op_fusion', \
                        help='Whether to enable Auto graph fusion feature. Default to True',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--micro_batch',
                        help='Set num for Auto Mirco Batch. Default close.',
                        type=int,
                        default=0)  #TODO: Defautl to True
    parser.add_argument('--adaptive_emb', \
                        help='Whether to enable Adaptive Embedding. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--dynamic_ev', \
                        help='Whether to enable Dynamic-dimension Embedding Variable. Default to False.',
                        type=boolean_string,
                        default=False)#TODO:enable
    parser.add_argument('--incremental_ckpt', \
                        help='Set time of save Incremental Checkpoint. Default 0 to close.',
                        type=int,
                        default=0)
    parser.add_argument('--workqueue', \
                        help='Whether to enable Work Queue. Default to False.',
                        type=boolean_string,
                        default=False)
    return parser

# parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    print(TF_CONFIG)
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
        print('TF_CONFIG ERROR')
        sys.exit()
    task_config = tf_config.get('task')
    task_type = task_config.get('type')
    task_index = task_config.get('index') + (1 if task_type == 'worker'
                                             and chief_hosts else 0)

    if task_type == 'chief':
        task_type = 'worker'

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.distribute.Server(cluster,
                                  job_name=task_type,
                                  task_index=task_index,
                                  protocol=args.protocol)
    if task_type == 'ps':
        server.join()
    elif task_type == 'worker':
        tf_config = {
            'ps_hosts': ps_hosts,
            'worker_hosts': worker_hosts,
            'type': task_type,
            'index': task_index,
            'is_chief': is_chief
        }
        tf_device = tf.device(
            tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % task_index,
                cluster=cluster))
        return tf_config, server, tf_device
    else:
        print("Task type or index error.")
        sys.exit()

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
    '''
    os.environ['START_STATISTIC_STEP'] = '100'
    os.environ['STOP_STATISTIC_STEP'] = '110'
    os.environ['MALLOC_CONF']= \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'

def check_stock_tf():
    import pkg_resources
    detailed_version = pkg_resources.get_distribution('Tensorflow').version
    return not ('deeprec' in detailed_version)

def check_DeepRec_features():
    return args.smartstaged or args.emb_fusion or args.op_fusion or args.micro_batch or args.bf16 or \
           args.ev or args.adaptive_emb or args.dynamic_ev or (args.optimizer == 'adamasync') or \
           args.incremental_ckpt  or args.workqueue

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    stock_tf = args.tf
    if not stock_tf and check_stock_tf() and check_DeepRec_features():
        raise ValueError('Stock Tensorflow does not support DeepRec features. '
                         'For Stock Tensorflow run the script with `--tf` argument.')

    if not args.tf:
        set_env_for_DeepRec()
    
    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        main()
    else:
        tf_config, server, tf_device = generate_cluster_info(TF_CONFIG)
        with tf_device:
            main(tf_config, server)

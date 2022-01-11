import time
import argparse
from numpy import dtype
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

CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ["clicked"]
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
HASH_BUCKET_SIZES = {
    'C1': 2500,
    'C2': 2000,
    'C3': 5000000,
    'C4': 1500000,
    'C5': 1000,
    'C6': 100,
    'C7': 20000,
    'C8': 4000,
    'C9': 20,
    'C10': 100000,
    'C11': 10000,
    'C12': 5000000,
    'C13': 40000,
    'C14': 100,
    'C15': 100,
    'C16': 3000000,
    'C17': 50,
    'C18': 10000,
    'C19': 4000,
    'C20': 20,
    'C21': 4000000,
    'C22': 100,
    'C23': 100,
    'C24': 250000,
    'C25': 400,
    'C26': 100000
}


def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def generate_input_data(filename, batch_size, num_epochs):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        cont_defaults = [[0.0] for i in range(1, 14)]
        cate_defaults = [[" "] for i in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cont_defaults + cate_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.shuffle(buffer_size=20000,
                              seed=2021)  # set seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(1)
    return dataset


def build_feature_cols():
    dense_column = []
    sparse_column = []
    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_COLUMNS:
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name,
                # hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                hash_bucket_size=10000,
                dtype=tf.string)

            sparse_column.append(
                tf.feature_column.embedding_column(categorical_column,
                                                   dimension=16,
                                                   combiner='mean'))
        else:
            column = tf.feature_column.numeric_column(column_name, shape=(1, ))
            dense_column.append(column)

    return dense_column, sparse_column


class DLRM():
    def __init__(self,
                 dense_column=None,
                 sparse_column=None,
                 mlp_bot=[512, 256, 64, 16],
                 mlp_top=[512, 256],
                 learning_rate=0.1,
                 inputs=None,
                 interaction_op='dot',
                 bf16=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        self.dense_column = dense_column
        self.sparse_column = sparse_column
        if not dense_column or not sparse_column:
            raise ValueError('Dense column or sparse column is not defined.')
        self.mlp_bot = mlp_bot
        self.mlp_top = mlp_top
        self.learning_rate = learning_rate
        self.input_layer_partitioner = input_layer_partitioner
        self.dense_layer_partitioner = dense_layer_partitioner

        self.feature = inputs[0]
        self.label = inputs[1]
        self.bf16 = bf16
        self._is_training = True

        self.interaction_op = interaction_op
        if self.interaction_op not in ['dot', 'cat']:
            print("Invaild interaction op, must be 'dot' or 'cat'.")
            sys.exit()

        self.predict = self.prediction()
        with tf.name_scope('head'):
            self.train_op, self.loss = self.optimizer()
            self.acc, self.acc_op = tf.metrics.accuracy(labels=self.label,
                                                        predictions=tf.round(
                                                            self.predict))
            self.auc, self.auc_op = tf.metrics.auc(labels=self.label,
                                                   predictions=self.predict,
                                                   num_thresholds=1000)
            tf.summary.scalar('eval_acc', self.acc)
            tf.summary.scalar('eval_auc', self.auc)

    def dot_op(self, features):
        batch_size = tf.shape(features)[0]
        matrixdot = tf.matmul(features, features, transpose_b=True)
        feature_dim = matrixdot.shape[-1]

        ones_mat = tf.ones_like(matrixdot)
        lower_tri_mat = ones_mat - tf.linalg.band_part(ones_mat, 0, -1)
        lower_tri_mask = tf.cast(lower_tri_mat, tf.bool)
        result = tf.boolean_mask(matrixdot, lower_tri_mask)
        output_dim = feature_dim * (feature_dim - 1) // 2

        return tf.reshape(result, (batch_size, output_dim))

    def prediction(self):
        # input dense feature and embedding of sparse features
        with tf.variable_scope('input_layer', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('dense_input_layer',
                                   partitioner=self.input_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                dense_inputs = tf.feature_column.input_layer(
                    self.feature, self.dense_column)
            with tf.variable_scope('sparse_input_layer', reuse=tf.AUTO_REUSE):
                column_tensors = {}
                sparse_inputs = tf.feature_column.input_layer(
                    self.feature,
                    self.sparse_column,
                    cols_to_output_tensors=column_tensors)

        # MLP behind dense inputs
        mlp_bot_scope = tf.variable_scope(
            'mlp_bot_layer',
            partitioner=self.dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with mlp_bot_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else mlp_bot_scope:
            if self.bf16:
                dense_inputs = tf.cast(dense_inputs, dtype=tf.bfloat16)

            for layer_id, num_hidden_units in enumerate(self.mlp_bot):
                with tf.variable_scope(
                        "mlp_bot_hiddenlayer_%d" % layer_id,
                        reuse=tf.AUTO_REUSE) as mlp_bot_hidden_layer_scope:
                    dense_inputs = tf.layers.dense(
                        dense_inputs,
                        units=num_hidden_units,
                        activation=tf.nn.relu,
                        name=mlp_bot_hidden_layer_scope)
                    dense_inputs = tf.layers.batch_normalization(
                        dense_inputs,
                        training=self._is_training,
                        trainable=True)
                    add_layer_summary(dense_inputs,
                                      mlp_bot_hidden_layer_scope.name)
            if self.bf16:
                dense_inputs = tf.cast(dense_inputs, dtype=tf.float32)

        #interaction_op
        if self.interaction_op == 'dot':
            # dot op
            with tf.variable_scope('Op_dot_layer', reuse=tf.AUTO_REUSE):
                net = [dense_inputs]
                for cols in self.sparse_column:
                    net.append(column_tensors[cols])
                net = tf.stack(net, axis=1)
                net = self.dot_op(net)
                net = tf.concat([dense_inputs, net], 1)
        elif self.interaction_op == 'cat':
            net = tf.concat([dense_inputs, sparse_inputs], 1)

        # top MLP before output
        if self.bf16:
            net = tf.cast(net, dtype=tf.bfloat16)
        mlp_top_scope = tf.variable_scope(
            'mlp_top_layer',
            partitioner=self.dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with mlp_top_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else mlp_top_scope:
            for layer_id, num_hidden_units in enumerate(self.mlp_top):
                with tf.variable_scope(
                        "mlp_top_hiddenlayer_%d" % layer_id,
                        reuse=tf.AUTO_REUSE) as mlp_top_hidden_layer_scope:
                    net = tf.layers.dense(net,
                                          units=num_hidden_units,
                                          activation=tf.nn.relu,
                                          name=mlp_top_hidden_layer_scope)

                add_layer_summary(net, mlp_top_hidden_layer_scope.name)

        if self.bf16:
            net = tf.cast(net, dtype=tf.float32)

        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE) as logits_scope:
            net = tf.layers.dense(net,
                                  units=1,
                                  activation=None,
                                  name=logits_scope)
            net = tf.math.sigmoid(net)

            add_layer_summary(net, logits_scope.name)

        return net

    def optimizer(self):
        bce_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.predict = tf.squeeze(self.predict)
        loss = tf.math.reduce_mean(bce_loss_func(self.label, self.predict))
        tf.summary.scalar('loss', loss)

        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)

        train_op = optimizer.minimize(loss, global_step=self.global_step)

        return train_op, loss


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output directory',
                        required=False)
    parser.add_argument('--deep_dropout',
                        help='Dropout regularization for deep model',
                        type=float,
                        default=0.0)
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.1)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument("--interaction_op",
                        type=str,
                        choices=["dot", "cat"],
                        default="cat")
    parser.add_argument("--protocol",
                        type=str,
                        choices=["grpc", "grpc++", "star_server"],
                        default="grpc")
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
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
    return parser


def main(tf_config=None, server=None):
    # check dataset
    print('Checking dataset')
    train_file = args.data_location + '/train.csv'
    test_file = args.data_location + '/eval.csv'

    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print(
            '------------------------------------------------------------------------------------------'
        )
        print(
            "train.csv or eval.csv does not exist in the given data_location. Please provide valid path"
        )
        print(
            '------------------------------------------------------------------------------------------'
        )
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of training dataset is {}".format(no_of_training_examples))
    print("Numbers of test dataset is {}".format(no_of_test_examples))

    # set params
    # set batch size & steps
    batch_size = args.batch_size
    if args.steps == 0:
        no_of_epochs = 10
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)

    # set fixed random seed
    tf.set_random_seed(2021)

    # set directory path
    model_dir = os.path.join(args.output_dir,
                             'model_DLRM_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline
    dense_column, sparse_column = build_feature_cols()
    train_dataset = generate_input_data(train_file, batch_size, no_of_epochs)
    test_dataset = generate_input_data(test_file, batch_size, 1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

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

    # create model
    model = DLRM(dense_column=dense_column,
                 sparse_column=sparse_column,
                 learning_rate=args.learning_rate,
                 bf16=args.bf16,
                 interaction_op=args.interaction_op,
                 inputs=next_element,
                 input_layer_partitioner=input_layer_partitioner,
                 dense_layer_partitioner=dense_layer_partitioner)

    sess_config = tf.ConfigProto()
    if args.inter:
        sess_config.inter_op_parallelism_threads = args.inter
    if args.intra:
        sess_config.intra_op_parallelism_threads = args.intra
    hooks = []

    if tf_config:
        hooks.append(tf.train.StopAtStepHook(last_step=train_steps))
        hooks.append(
            tf.train.LoggingTensorHook(
                {
                    'steps': model.global_step,
                    'loss': model.loss
                },
                every_n_iter=100))

        scaffold = tf.train.Scaffold(
            local_init_op=tf.group(tf.global_variables_initializer(
            ), tf.local_variables_initializer(), train_init_op))

        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=tf_config['is_chief'],
                checkpoint_dir=checkpoint_dir,
                scaffold=scaffold,
                hooks=hooks,
                # save_checkpoint_steps=args.save_steps,
                log_step_count_steps=100,
                config=sess_config) as sess:
            while not sess.should_stop():
                _, train_loss = sess.run([model.train_op, model.loss])
    else:
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=args.keep_checkpoint_max)
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # train model
            sess.run(train_init_op)

            start = time.perf_counter()
            for _in in range(0, train_steps):
                if args.save_steps > 0 and (_in % args.save_steps == 0
                                            or _in == train_steps - 1):
                    _, train_loss, events = sess.run(
                        [model.train_op, model.loss, merged])
                    writer.add_summary(events, _in)
                    checkpoint_path = saver.save(sess,
                                                 save_path=os.path.join(
                                                     checkpoint_dir,
                                                     'dlrm-checkpoint'),
                                                 global_step=_in)
                    print("Save checkpoint to %s" % checkpoint_path)
                elif (args.timeline > 0 and _in % args.timeline == 0):
                    _, train_loss = sess.run([model.train_op, model.loss],
                                             options=options,
                                             run_metadata=run_metadata)
                    fetched_timeline = timeline.Timeline(
                        run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format(
                    )
                    print("Save timeline to %s" % checkpoint_dir)
                    with open(
                            os.path.join(checkpoint_dir,
                                         'timeline-%d.json' % _in), 'w') as f:
                        f.write(chrome_trace)
                else:
                    _, train_loss = sess.run([model.train_op, model.loss])

                # print training loss and time cost
                if (_in % 100 == 0 or _in == train_steps - 1):
                    end = time.perf_counter()
                    cost_time = end - start
                    global_step_sec = (100 if _in % 100 == 0 else train_steps -
                                       1 % 100) / cost_time
                    print("global_step/sec: %0.4f" % global_step_sec)
                    print("loss = {}, steps = {}, cost time = {:0.2f}s".format(
                        train_loss, _in, cost_time))
                    start = time.perf_counter()

            # eval model
            if not args.no_eval:
                model._is_training = False
                writer = tf.summary.FileWriter(
                    os.path.join(checkpoint_dir, 'eval'))

                sess.run(test_init_op)
                sess.run(tf.local_variables_initializer())
                for _in in range(1, test_steps + 1):
                    if (_in != test_steps):
                        sess.run(
                            [model.acc, model.acc_op, model.auc, model.auc_op])
                        if (_in % 1000 == 0):
                            print("Evaluation complate:[{}/{}]".format(
                                _in, test_steps))
                    else:
                        _, eval_acc, _, eval_auc, events = sess.run([
                            model.acc, model.acc_op, model.auc, model.auc_op,
                            merged
                        ])
                        writer.add_summary(events, _in)
                        print("Evaluation complate:[{}/{}]".format(
                            _in, test_steps))
                        print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        main()
    else:
        print(TF_CONFIG)
        tf_config = json.loads(TF_CONFIG)
        cluster_config = tf_config.get('cluster')
        ps_hosts = []
        worker_hosts = []
        chief_hosts = []
        for key, value in cluster_config.items():
            if "ps" == key:
                ps_hosts = value
            elif "worker" == key:
                worker_hosts = value
            elif "chief" == key:
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
        cluster = tf.train.ClusterSpec({
            "ps": ps_hosts,
            "worker": worker_hosts
        })
        server = tf.distribute.Server(cluster,
                                      job_name=task_type,
                                      task_index=task_index,
                                      protocol=args.protocol)
        if task_type == 'ps':
            server.join()
        elif task_type == 'worker':
            with tf.device(
                    tf.train.replica_device_setter(
                        worker_device="/job:worker/task:%d" % task_index,
                        cluster=cluster)):
                main(tf_config={
                    'ps_hosts': ps_hosts,
                    'worker_hosts': worker_hosts,
                    'type': task_type,
                    'index': task_index,
                    'is_chief': is_chief
                },
                     server=server)
        else:
            print("Task type or index error.")
            sys.exit()
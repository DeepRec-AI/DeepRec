import time
import argparse
import tensorflow as tf
import os
import sys
import math
import collections

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
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
EMBEDDING_DIMENSIONS = {
    'C1': 64,
    'C2': 64,
    'C3': 128,
    'C4': 128,
    'C5': 64,
    'C6': 64,
    'C7': 64,
    'C8': 64,
    'C9': 64,
    'C10': 128,
    'C11': 64,
    'C12': 128,
    'C13': 64,
    'C14': 64,
    'C15': 64,
    'C16': 128,
    'C17': 64,
    'C18': 64,
    'C19': 64,
    'C20': 64,
    'C21': 128,
    'C22': 64,
    'C23': 64,
    'C24': 128,
    'C25': 64,
    'C26': 128
}


def add_layer_summary(value, tag):
    tf.compat.v1.summary.scalar('%s/fraction_of_zero_values' % tag,
                                tf.nn.zero_fraction(value))
    tf.compat.v1.summary.histogram('%s/activation' % tag, value)


def generate_input_data(filename, batch_size, num_epochs):
    def parse_csv(value):
        tf.compat.v1.logging.info('Parsing {}'.format(filename))
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


def build_feature_cols(train_file_path, test_file_path):
    # Statistics of Kaggle's Criteo Dataset has been calculated in advance to save time
    print('****Computing statistics of train dataset*****')
    # with open(train_file_path, 'r') as f, open(test_file_path, 'r') as f1:
    #     nums = [line.strip('\n').split(',') for line in f.readlines()
    #             ] + [line.strip('\n').split(',') for line in f1.readlines()]
    #     numpy_arr = np.array(nums)
    #     mins_list, max_list, range_list = [], [], []
    #     for i in range(len(TRAIN_DATA_COLUMNS)):
    #         if TRAIN_DATA_COLUMNS[i] in CONTINUOUS_COLUMNS:
    #             col_min = numpy_arr[:, i].astype(np.float32).min()
    #             col_max = numpy_arr[:, i].astype(np.float32).max()
    #             mins_list.append(col_min)
    #             max_list.append(col_max)
    #             range_list.append(col_max - col_min)
    mins_list = [
        0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    range_list = [
        1539.0, 22069.0, 65535.0, 561.0, 2655388.0, 233523.0, 26297.0, 5106.0,
        24376.0, 9.0, 181.0, 1807.0, 6879.0
    ]

    def make_minmaxscaler(min, range):
        def minmaxscaler(col):
            return (col - min) / range

        return minmaxscaler

    dense_column = []
    sparse_column = []
    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_COLUMNS:
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name,
                # hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                hash_bucket_size=1000,
                dtype=tf.string)

            sparse_column.append(
                tf.feature_column.embedding_column(
                    categorical_column,
                    # dimension=EMBEDDING_DIMENSIONS[column_name],
                    dimension=16,
                    combiner='mean'))
        else:
            normalizer_fn = None
            i = CONTINUOUS_COLUMNS.index(column_name)
            col_min = mins_list[i]
            col_range = range_list[i]
            normalizer_fn = make_minmaxscaler(col_min, col_range)
            # column = tf.feature_column.numeric_column(column_name, shape=(1, ))
            column = tf.feature_column.numeric_column(
                column_name, normalizer_fn=normalizer_fn, shape=(1, ))
            dense_column.append(column)

    return dense_column, sparse_column


class DLRM():
    def __init__(self,
                 dense_column=None,
                 sparse_column=None,
                 mlp_bot=[512, 256, 64, 16],
                 mlp_top=[512, 256],
                 learning_rate=0.1,
                 inputs=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        self.dense_column = dense_column
        self.sparse_column = sparse_column
        if not dense_column or not sparse_column:
            raise ValueError('Dense column or sparse column is not defined.')
        self.mlp_bot = mlp_bot
        self.mlp_top = mlp_top
        self.learning_rate = learning_rate
        self.feature = inputs[0]
        self.label = inputs[1]

        self.predict = self.prediction()
        self.train_op, self.loss = self.optimizor()
        self.acc, self.acc_op = tf.compat.v1.metrics.accuracy(
            labels=self.label, predictions=self.predict)
        self.auc, self.auc_op = tf.compat.v1.metrics.auc(
            labels=self.label, predictions=self.predict, num_thresholds=1000)
        tf.compat.v1.summary.scalar('eval_acc', self.acc)
        tf.compat.v1.summary.scalar('eval_auc', self.auc)

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
        with tf.compat.v1.variable_scope('input_layer',
                                         reuse=tf.compat.v1.AUTO_REUSE):
            with tf.compat.v1.variable_scope('dense_input_layer',
                                             reuse=tf.compat.v1.AUTO_REUSE):
                dense_inputs = tf.compat.v1.feature_column.input_layer(
                    self.feature, self.dense_column)
            with tf.compat.v1.variable_scope('sparse_input_layer',
                                             reuse=tf.compat.v1.AUTO_REUSE):
                column_tensors = {}
                sparse_inputs = tf.compat.v1.feature_column.input_layer(
                    self.feature,
                    self.sparse_column,
                    cols_to_output_tensors=column_tensors)

        # MLP behind dense inputs
        with tf.compat.v1.variable_scope('mlp_bot_layer',
                                         reuse=tf.compat.v1.AUTO_REUSE):
            for layer_id, num_hidden_units in enumerate(self.mlp_bot):
                with tf.compat.v1.variable_scope(
                        "mlp_bot_hiddenlayer_%d" % layer_id,
                        reuse=tf.compat.v1.AUTO_REUSE
                ) as mlp_bot_hidden_layer_scope:
                    dense_inputs = tf.layers.dense(
                        dense_inputs,
                        units=num_hidden_units,
                        activation=tf.nn.relu,
                        name=mlp_bot_hidden_layer_scope)
                    add_layer_summary(dense_inputs,
                                      mlp_bot_hidden_layer_scope.name)

        # dot op
        with tf.compat.v1.variable_scope('Op_dot_layer',
                                         reuse=tf.compat.v1.AUTO_REUSE):
            net = tf.expand_dims(dense_inputs, 1)
            for cols in self.sparse_column:
                net = tf.concat(
                    [net, tf.expand_dims(column_tensors[cols], 1)], 1)
            net = self.dot_op(net)
            net = tf.concat([dense_inputs, net], 1)

        # top MLP before output
        with tf.compat.v1.variable_scope('mlp_top_layer',
                                         reuse=tf.compat.v1.AUTO_REUSE):
            for layer_id, num_hidden_units in enumerate(self.mlp_top):
                with tf.compat.v1.variable_scope(
                        "mlp_top_hiddenlayer_%d" % layer_id,
                        reuse=tf.compat.v1.AUTO_REUSE) as mlp_top_hidden_layer_scope:
                    net = tf.layers.dense(net,
                                          units=num_hidden_units,
                                          activation=tf.nn.relu,
                                          name=mlp_top_hidden_layer_scope)

                add_layer_summary(net, mlp_top_hidden_layer_scope.name)

            with tf.compat.v1.variable_scope("mlp_top_hiddenlayer_last",
                                             reuse=tf.compat.v1.AUTO_REUSE
                                             ) as mlp_top_hidden_layer_scope:
                net = tf.layers.dense(net,
                                      units=1,
                                      activation='sigmoid',
                                      name=mlp_top_hidden_layer_scope)

                add_layer_summary(net, mlp_top_hidden_layer_scope.name)

        return net

    def optimizor(self):
        bce_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.predict = tf.squeeze(self.predict)
        loss = tf.math.reduce_mean(bce_loss_func(self.label, self.predict))
        tf.compat.v1.summary.scalar('loss', loss)

        global_step = tf.compat.v1.train.get_global_step()
        optimizorr = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        train_op = optimizorr.minimize(loss, global_step=global_step)

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
                        default=500)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

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
    if (args.save_steps <= 0):
        print(
            "Save steps should be a positive integer. Please provide valid value"
        )
        sys.exit()

    # set fixed random seed
    tf.compat.v1.set_random_seed(2021)

    # set directory path
    model_dir = os.path.join(args.output_dir,
                             'model_DLRM_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline
    dense_column, sparse_column = build_feature_cols(train_file, test_file)
    train_dataset = generate_input_data(train_file, batch_size, no_of_epochs)
    test_dataset = generate_input_data(test_file, batch_size, 1)

    iterator = tf.compat.v1.data.Iterator.from_structure(
        train_dataset.output_types, test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create model
    model = DLRM(dense_column=dense_column,
                 sparse_column=sparse_column,
                 learning_rate=args.learning_rate,
                 inputs=next_element)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        merged = tf.compat.v1.summary.merge_all()
        writer = tf.compat.v1.summary.FileWriter(checkpoint_dir, sess.graph)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                         max_to_keep=args.keep_checkpoint_max)

        # train model
        sess.run(train_init_op)

        # save first step checkpoint
        _, train_loss, events = sess.run([model.train_op, model.loss, merged])
        writer.add_summary(events, 0)
        print("loss = {}, steps = 0".format(train_loss))
        checkpoint_path = saver.save(sess,
                                     save_path=os.path.join(
                                         checkpoint_dir, 'dlrm-checkpoint'),
                                     global_step=0)
        print("Save checkpoint to %s" % checkpoint_path)

        start = time.perf_counter()
        for _in in range(1, train_steps):
            _, train_loss, events = sess.run(
                [model.train_op, model.loss, merged])

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

            # save tensorboard events
            if ((args.save_steps > 0 and _in % args.save_steps == 0)
                    or _in == train_steps - 1):
                writer.add_summary(events, _in)
                checkpoint_path = saver.save(sess,
                                             save_path=os.path.join(
                                                 checkpoint_dir,
                                                 'dlrm-checkpoint'),
                                             global_step=_in)
                print("Save checkpoint to %s" % checkpoint_path)

        # eval model
        writer = tf.compat.v1.summary.FileWriter(
            os.path.join(checkpoint_dir, 'eval'))

        sess.run(test_init_op)
        for _in in range(1, test_steps + 1):
            eval_acc, _, eval_auc, _, events = sess.run(
                [model.acc, model.acc_op, model.auc, model.auc_op, merged])
            if (_in % 1000 == 0):
                print("Evaluation complate:[{}/{}]".format(_in, test_steps))
            if (_in == test_steps):
                writer.add_summary(events, _in)
                print("Evaluation complate:[{}/{}]".format(_in, test_steps))
                print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))

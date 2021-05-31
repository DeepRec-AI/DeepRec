"""Tests for Structured Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.structured_model.python.core import *
from tensorflow.contrib.structured_model.python.utils import *
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
# Helpers for creating Example objects
example = example_pb2.Example
feature = feature_pb2.Feature
features = lambda d: feature_pb2.Features(feature=d)
bytes_feature = lambda v: feature(bytes_list=feature_pb2.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=feature_pb2.Int64List(value=v))
float_feature = lambda v: feature(float_list=feature_pb2.FloatList(value=v))
# Helpers for creating SequenceExample objects
feature_list = lambda l: feature_pb2.FeatureList(feature=l)
feature_lists = lambda d: feature_pb2.FeatureLists(feature_list=d)
sequence_example = example_pb2.SequenceExample

json_string = '{ "feature_columns": [ { "feature_name": "user_id", "feature_type": "raw_feature", "value_type": "int64", "expression": "user:f1" }, { "feature_name": "f1", "feature_type": "id_feature", "value_type": "string", "hash_bucket_size": 1000, "embedding_dimension": 8, "expression": "item:f2" }, { "feature_name": "seq_label", "feature_type": "structured_sequence_feature", "delim": ";", "length": 5, "features": [ { "feature_name": "click", "feature_type": "id_feature", "value_type": "float", "expression": "item:seq_f1" }, { "feature_name": "pay", "feature_type": "id_feature", "value_type": "string", "hash_bucket_size": 1000, "embedding_dimension": 8, "expression": "item:seq_f1" } ] } ] }'

class StructuredModelUtilTest(test.TestCase):
  def testUtilsParseExmaple(self):

    sex = example(features=features({"user_id": int64_feature([123]),
                                     "seq_label_pay_0": bytes_feature([b'b0s0']),
                                     "seq_label_pay_1": bytes_feature([b'b0s1']),
                                    }))
    sex2 = example(features=features({"user_id": int64_feature([234]),
                                     "seq_label_pay_0": bytes_feature([b'b1s0']),
                                     "seq_label_pay_1": bytes_feature([b'b1s1']),
                                    }))
    ser = sex.SerializeToString()
    ser2 = sex2.SerializeToString()
    ser = ops.convert_to_tensor([ser, ser2])

    with open('fg_test.json', 'w') as f:
      f.write(json_string)
    features_ = parse_example(ser, 'fg_test.json')
    user_seq_layer = feature_column_ops.input_from_feature_columns(
            features_,
            [feature_column.embedding_column(feature_column.sparse_column_with_hash_bucket('seq_label_pay', hash_bucket_size=100),
                                             dimension=8, initializer=init_ops.constant_initializer(200))])
    print("user_seq_layer",user_seq_layer)
    items = array_ops.split(user_seq_layer, 2, axis=0)  # [?, f_num*f_embedding]*30
    items_stack = array_ops.stack(values=items, axis=1)

    with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(variables.local_variables_initializer())
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(coord=coord, sess=sess)

        ret = sess.run([features_, items_stack, user_seq_layer])
        print(ret)
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
      test.main()

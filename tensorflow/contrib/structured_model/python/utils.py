from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import collections
import json

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import tf_logging as logging

def generate_feature_spec_from_json(file_path):
    def generate_feature_by_type(feature_type, value_type):
      if feature_type == 'raw_feature':
        return parsing_ops.FixedLenFeature([], dtypes.as_dtype(value_type))
      elif feature_type == 'id_feature':
        return parsing_ops.VarLenFeature(dtypes.as_dtype(value_type))

    with open(file_path) as f:
      conf = json.load(f)
    features = {}
    for feature_col in conf['feature_columns']:
      f_name = feature_col['feature_name']
      f_type = feature_col['feature_type']
      if f_type == 'sequence_feature' or f_type == 'structured_sequence_feature':
        seq_features = feature_col['features']
        for feature in seq_features:
          seq_f_name = '_'.join([f_name, feature['feature_name']])
          f_type = feature['feature_type']
          for i in range(feature_col['length']):
            features['_'.join([seq_f_name, str(i)])] = generate_feature_by_type(f_type, feature['value_type'])
      else:
        features[f_name] = generate_feature_by_type(f_type, feature_col['value_type'])
    return features

def parse_example(example, file_path):
    parsed_examples = parsing_ops.parse_example(example,
                                     features=generate_feature_spec_from_json(file_path))

    with open(file_path) as f:
      conf = json.load(f)
    features = {}
    for feature_col in conf['feature_columns']:
      f_type = feature_col['feature_type']
      if f_type == 'sequence_feature' or f_type == 'structured_sequence_feature':
        seq_features = feature_col['features']
        f_name = feature_col['feature_name']
        for feature in seq_features:
          seq_f_name = '_'.join([f_name, feature['feature_name']])
          f_type = feature['feature_type']
          f_list = []
          for i in range(feature_col['length']):
            seq_f_name_index = '_'.join([seq_f_name, str(i)])
            f_list.append(parsed_examples[seq_f_name_index])
            del parsed_examples[seq_f_name_index]
          parsed_examples[seq_f_name] = sparse_ops.sparse_concat(
                  sp_inputs=f_list,
                  axis=0,
                  expand_nonconcat_dim=True)
    return parsed_examples

if __name__ == "__main__":
    generate_feature_spec_from_json('fg_test.json')

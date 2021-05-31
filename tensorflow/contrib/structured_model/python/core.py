from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import collections

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import tf_logging as logging

_structured_model = None
__SEQUENCE_LABEL__ = "__seq_label__"

def get_structured_model():
  return _structured_model

def enable_structured_model_mode(fg, mc, features, label,
                                 user_tensor, item_tensor, query_tensor=None,
                                 scope=None):
  global _structured_model
  _structured_model = StructuredModel(fg, mc, features, label,
                                      user_tensor, item_tensor, query_tensor,
                                      scope=scope)
  logging.info("enable structured_model mode, %s", _structured_model)

def find_boundery_tensors(user_ops, item_ops):
  queue_item = collections.deque()
  queue_item.extend(item_ops)
  item_sets = set()
  while queue_item:
    op = queue_item.popleft()
    for t in op.outputs:
      item_sets = (item_sets|set(t.consumers()))
      queue_item.extend(t.consumers())

  queue_user = collections.deque()
  queue_user.extend(user_ops)
  user_sets = set()
  boundery_tensor_sets = set()
  while queue_user:
    op = queue_user.popleft()
    for t in op.outputs:
      for op2 in t.consumers():
        if op2 in user_sets:
          continue
        if op2 in item_sets:
          boundery_tensor_sets.add(t)
        else:
          user_sets.add(op2)
          queue_user.append(op2)

  logging.info("boundery_tensor_sets: %s", boundery_tensor_sets)
  return user_sets, item_sets, boundery_tensor_sets

def add_tile_op(boundery_tensor_sets, fg, seq_mask_reshaped, user_sets):
  for t in boundery_tensor_sets:
    if not t.consumers():
      continue
    user_expand = array_ops.expand_dims(t, 1)
    user_tiled = array_ops.tile(user_expand, [1, fg.item_seq_length, 1])
    user_2d = array_ops.reshape(user_tiled, [-1, user_tiled.get_shape()[2]])
    seq_user_input = array_ops.boolean_mask(user_2d, seq_mask_reshaped)
    logging.info("add_tile_op: %s, %s", t, t.consumers())
    t_ops = copy.copy(t.consumers())
    for op in t_ops:
      if op is user_expand.op or op in user_sets:
        continue
      for index, input_t in enumerate(op.inputs):
        if input_t is t:
          op._update_input(index, seq_user_input)
          logging.info("add_tile_op detail: %s, %s, %s", op, index, seq_user_input)

def add_split_op(fg, mc_columns, features, item_tensor,
                 seq_mask_reshaped, scope):
  if not item_tensor.consumers():
    return
  seq_features = fg.sequence_features(features, mc_columns,
                                      fg.item_seq_length,
                                      fg.item_seq_name,
                                      False)
  column = fg.feature_columns_from_name(mc_columns)
  from tensorflow.contrib import layers
  seq_layer = layers.input_from_feature_columns(seq_features,
                                                column, default_id=0,
                                                scope=scope)

  seq = array_ops.split(seq_layer, fg.item_seq_length, axis=0)
  seq_stack = array_ops.stack(values=seq, axis=1)
  seq_2d = array_ops.reshape(seq_stack, [-1, seq_stack.get_shape()[2]])
  seq_item = array_ops.boolean_mask(seq_2d, seq_mask_reshaped)

  t = item_tensor
  logging.info("add_split_op: %s, %s", t, t.consumers())
  t_ops = copy.copy(t.consumers())
  for op in t_ops:
    for index, input_t in enumerate(op.inputs):
      if input_t is t:
        op._update_input(index, seq_item)
        logging.info("add_split_op detail: %s, %s, %s", op, index, seq_item)

def add_label_op(fg, features, label_tensor, seq_mask_reshaped):
  if not label_tensor.consumers():
    return
  raw_seq_label = array_ops.reshape(sparse_ops.sparse_tensor_to_dense(
                                      features['itm_click_seq'],
                                      default_value="0"),
                                    [-1, 1])
  default_values = [constant_op.constant([0.0], dtype=dtypes.float32)
                      for i in range(0, fg.item_seq_length)]
  seq_labels = parsing_ops.decode_csv(raw_seq_label,
                                      record_defaults=default_values,
                                      field_delim=';')
  seq_labels = array_ops.transpose(seq_labels)
  item_seq_label_2d = array_ops.reshape(seq_labels, [-1, 1])
  seq_label = array_ops.boolean_mask(item_seq_label_2d, seq_mask_reshaped)
  ops.add_to_collection(__SEQUENCE_LABEL__, seq_label)
  logging.info("add_label_op: %s, %s", label_tensor, label_tensor.consumers())
  t_ops = copy.copy(label_tensor.consumers())
  for op in t_ops:
    for index, input_t in enumerate(op.inputs):
      if input_t is label_tensor:
        op._update_input(index, seq_label)
        logging.info("add_label_op detail: %s, %s, %s", op, index, seq_label)

class StructuredModel(object):

  def __init__(self, fg, mc, features, label,
               user_tensor, item_tensor, query_tensor=None,
               scope=None):
    self.fg = fg
    self.mc = mc
    self.features = features
    self.label = label
    self.user_tensor = user_tensor
    self.item_tensor = item_tensor
    self.query_tensor = query_tensor
    self.scope = scope

  def graph_transform(self):
    """
    Graph transform
    ====> Behavior
    first time call graph_transform()
      Graph:
        user_tenosr_subGraph ----> user_net ------------\
                                                         -----> other_net
        non-user_tenosr_subGraph ----> non-user_net ----/

                                       ||
                                       ||
                                       ||
                                       \/

        user_tenosr_subGraph ----> user_net ----> subGraph1 ----\
                                                                 ----> other_net
        subGraph2 ----> non-user_net ---------------------------/
      Detail:
        a .build 'sequence_mask' subGraph
        b. find boundery_tensor which collect user-subGraph and non-user-subGraph
        c. build subGraph_1 after boundery_tensor
        d. build subGraph_2 for non-user tensor(item/label) to replace former graph
    second time call graph_transform()
        Redo d. build subGraph_2 for non-user tensor(item/label) to replace former graph
        ATTENTION: DONNOT do c. because boundery_tensor is in backprop subGraph

    ====> Build Graph
    (build graph start)
    build_input()
    build_forward()
    build_loss()
    (call StructuredModel.graph_transform for forward graph transform)
    build_backprop()    (create optimizer, and compute grad. & apply grad.)
    build_other_graph()    (add summary info.)
    (call StructuredModel.graph_transform )
    (build graph end)
    """
    logging.info("graph_transform: start")
    user_op = self.user_tensor.op
    item_op = self.item_tensor.op
    query_op = self.query_tensor.op

    seq_mask_reshaped = ops.get_collection("_seq_mask_reshaped")
    if not seq_mask_reshaped:
      itm_seq_length = self.features["itm_seq_length"]
      item_seq_mask = array_ops.sequence_mask(array_ops.reshape(itm_seq_length, [-1]),
                                              self.fg.item_seq_length)
      seq_mask_reshaped = array_ops.reshape(item_seq_mask, [-1])
      ops.add_to_collection("_seq_mask_reshaped", seq_mask_reshaped)

      user_op_sets, _, boundery_tensor_sets = find_boundery_tensors(
          user_ops=[user_op],
          item_ops=[item_op] + [query_op])
      add_tile_op(boundery_tensor_sets, self.fg, seq_mask_reshaped, user_op_sets)

    seq_mask_reshaped = ops.get_collection("_seq_mask_reshaped")[0]
    add_split_op(self.fg, self.mc.item_columns, self.features,
                 self.item_tensor, seq_mask_reshaped, scope=self.scope)
    if query_op:
      add_split_op(self.fg, self.mc.query_columns, self.features,
                   self.query_tensor, seq_mask_reshaped, scope=self.scope)

    add_label_op(self.fg, self.features, self.label, seq_mask_reshaped)
    logging.info("graph_transform: end")


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import collections

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_util, sparse_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

_sample_awared_graph = None

def get_sample_awared_graph():
  return _sample_awared_graph

@tf_export('graph_optimizer.enable_sample_awared_graph_compression')
def enable_sample_awared_graph_compression(user_tensors, item_tensors, item_size):
  global _sample_awared_graph
  _sample_awared_graph = SampleAwaredGraph(user_tensors, item_tensors, item_size)

def find_boundery_tensors(user_ops, item_ops):
  queue_item = collections.deque()
  queue_item.extend([x for x, _ in item_ops])
  queue_item_back = collections.deque()
  item_sets = set()
  processed_item = set()
  while queue_item:
    op = queue_item.popleft()
    if control_flow_util.IsInWhileLoop(op):
      input_ops = []
      for t in op.inputs:
        input_ops.append(t.op)
      queue_item_back.extend(input_ops)
    if op in processed_item:
      continue
    processed_item.add(op)
    for t in op.outputs:
      item_sets = (item_sets|set(t.consumers()))
      queue_item.extend(t.consumers())
  processed_item = set()
  while queue_item_back:
    op = queue_item_back.popleft()
    if op in processed_item:
      continue
    if control_flow_util.IsInWhileLoop(op):
      input_ops = []
      for t in op.inputs:
        input_ops.append(t.op)
      for t in op.outputs:
        input_ops.append(t.op)
      queue_item_back.extend(input_ops)
      item_sets.add(op)
      processed_item.add(op)

  queue_user = collections.deque()
  queue_user.extend(user_ops)
  user_sets = set([x for x, _ in user_ops])
  boundery_tensor_sets = set()
  while queue_user:
    op, is_sparse_tensor = queue_user.popleft()
    for t in op.outputs:
      for op2 in t.consumers():
        if op2 in user_sets:
          continue
        if op2 in item_sets:
          boundery_tensor_sets.add((t, is_sparse_tensor))
        else:
          user_sets.add(op2)
          queue_user.append((op2, False if op2.type == "SparseCross" else is_sparse_tensor))
  logging.info("[SampleAwaredGraphCompression] boundery_tensor_sets: %s", boundery_tensor_sets)
  return user_sets, item_sets, boundery_tensor_sets

def is_shape_op(op):
  return op.type == "Shape" or op.type == "ShapeN" or op.type == "Rank" or op.type == "Size"

def add_tile_op(boundery_tensor_sets, item_seq_length, user_sets, seq_mask_reshaped=None):
  # For training process, we cannot known the exact size of a item sequence,
  # so we need a masking. However, we do not need masking for serving as we
  # can safely assume the maximum size of the query items are item_seq_length.
  tiled_num = 0
  for t, is_sparse_tensor in boundery_tensor_sets:
    if not t.consumers():
      continue
    # only add tiles for those with batched tensors (dynamic shaped [?, ...])
    # as some constant operations, such as reshape, should not be tiled
    if len(t.get_shape().as_list()) > 0 and not t.get_shape().as_list()[0]:
      with ops.colocate_with(t.op):
        if is_sparse_tensor:
          range_tensor = math_ops.range(array_ops.squeeze(item_seq_length), dtype=dtypes.int64)
          user_expand = array_ops.tile(t, array_ops.concat([item_seq_length, [1]], axis=0))
          zero_tensor = array_ops.zeros_like(range_tensor, dtypes.int64)
          stack = array_ops.stack([range_tensor, zero_tensor], axis=1)
          repeat_tensor = array_ops.repeat(array_ops.expand_dims(array_ops.shape(t)[0], axis=0), item_seq_length, axis=0)
          repeat_tensor = array_ops.repeat(stack, repeat_tensor, axis=0)
          seq_user_input = repeat_tensor + user_expand
        else:
          user_expand = array_ops.expand_dims(t, 1)
          tile_shape = array_ops.concat([[1], item_seq_length, [1 for i in range(len(t.get_shape()[1:]))]], axis=0)
          user_tiled = array_ops.tile(user_expand, tile_shape)
          reshape_shape = [-1]
          if len(user_tiled.get_shape()) > 2:
            user_tiled_shape = array_ops.shape(user_tiled)
            reshape_shape = array_ops.concat([math_ops.reduce_prod(user_tiled_shape[:2],keepdims=True), user_tiled_shape[2:]], 0)
          seq_user_input = array_ops.reshape(user_tiled, reshape_shape)
        if seq_mask_reshaped:
          seq_user_input = array_ops.boolean_mask(seq_user_input, seq_mask_reshaped)
        t_ops = copy.copy(t.consumers())
        for op in t_ops:
          if not is_shape_op(op):
            if op is user_expand.op or op in user_sets:
              continue
          for index, input_t in enumerate(op.inputs):
            if input_t is t:
              op._update_input(index, seq_user_input)
        tiled_num += 1
  logging.info("[SampleAwaredGraphCompression] add %d TileOp", tiled_num)

class SampleAwaredGraph(object):
  # user_tensors: list of Tensor or SparseTensor
  # item_tensors: list of packed tensor
  def __init__(self, user_tensors, item_tensors, item_seq_length):
    self.user_tensors = user_tensors
    self.item_tensors = item_tensors
    self.item_seq_length = item_seq_length

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
        a. build 'sequence_mask' subGraph
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
    (call sample_awared_graph.graph_transform for forward graph transform)
    # TODO: support training
    # build_backprop()    (create optimizer, and compute grad. & apply grad.)
    # build_other_graph()    (add summary info.)
    # (call sample_awared_graph.graph_transform )
    # (build graph end)
    """
    user_ops = []
    item_ops = []
    for t in self.user_tensors:
      if isinstance(t, ops.Tensor):
        user_ops.append((t.op, False))
      elif isinstance(t, sparse_tensor.SparseTensor):
        user_ops.append((t.indices.op, True))
        user_ops.append((t.values.op, False))
      else:
        raise ValueError("user_tensors mustbe Tensor or SparseTensor.")
    for t in self.item_tensors:
      if isinstance(t, ops.Tensor):
        item_ops.append((t.op, False))
      elif isinstance(t, sparse_tensor.SparseTensor):
        item_ops.append((t.indices.op, True))
        item_ops.append((t.values.op, False))
      else:
        raise ValueError("item_tensors mustbe Tensor or SparseTensor.")

    # subGraph_1
    user_op_sets, _, boundery_tensor_sets = find_boundery_tensors(
        user_ops=user_ops,
        item_ops=item_ops)
    add_tile_op(boundery_tensor_sets, self.item_seq_length, user_op_sets)

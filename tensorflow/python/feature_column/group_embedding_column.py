from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

_global_fusion_embedding_scope = []
_group_embedding_tensor = dict()

def _global_group_embedding_scope_list():
    global _global_fusion_embedding_scope
    return _global_fusion_embedding_scope

def _current_group_embedding_scope():
    global _global_fusion_embedding_scope
    return None if len(_global_fusion_embedding_scope) == 0 else _global_fusion_embedding_scope[-1]

def _get_global_group_embedding_scope(embedding_columns,
                                      builder=None,
                                      weight_collections=None,
                                      trainable=True):
    global _group_embedding_tensor
    global _global_fusion_embedding_scope
    fused_scope = _global_fusion_embedding_scope[-1]
    filter_ec, admitted_ec = [], []
    for ec in embedding_columns:
        if ec in _group_embedding_tensor:
            filter_ec.append(ec)
        else:
            admitted_ec.append(ec)
    fused_output, sequence_lengths = fused_scope._get_dense_tensor(
        admitted_ec, builder, weight_collections, trainable)

    for ec, output, sequence_length in zip(admitted_ec, fused_output, sequence_lengths): #Ordered
        _group_embedding_tensor[ec] = (output, sequence_length)
    return _group_embedding_tensor


class GroupEmbeddingScopeBase(object):
    def __init__(self, name=None, params_num_per_group=sys.maxsize):
        self.name = name
        self.params_num_per_group = params_num_per_group
        self.embedding_columns = []

    def add_column(self, embedding_column):
        raise NotImplementedError("Valid EmbeddingColumn should be "
                                  "specified by successor.")

    def _get_dense_tensor(self, admitted_ec, inputs, weight_collections=None, trainable=None, is_sequence=False):
        raise NotImplementedError("should be implement in successor.")

    def get_dense_tensor(self, transformation_cache, state_manager):
        raise NotImplementedError("should be implement in successor.")

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_global_fusion_embedding_scope = []
_group_id = 0

def _global_group_embedding_scope_list():
    global _global_fusion_embedding_scope
    return _global_fusion_embedding_scope

def _current_group_embedding_scope():
    global _global_fusion_embedding_scope
    return None if len(_global_fusion_embedding_scope) == 0 else _global_fusion_embedding_scope[-1]

def _current_group_id():
    global _group_id
    return _group_id

class GroupEmbeddingScopeBase(object):
    def __init__(self, name=None):
        self.name = name
        self.embedding_columns = []

    def add_column(self, embedding_column):
        raise NotImplementedError("Valid EmbeddingColumn should be "
                                  "specified by successor.")

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        raise NotImplementedError("should be implement in successor.")

    def get_dense_tensor(self, transformation_cache, state_manager):
        raise NotImplementedError("should be implement in successor.")

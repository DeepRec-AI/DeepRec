from tensorflow.python.framework import ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_fused_embedding_ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers.feature_column_ops import check_feature_columns
from tensorflow.contrib.layers.python.layers.feature_column_ops import _Transformer
from tensorflow.contrib.layers.python.layers import feature_column as fc


def embeddings_from_arguments_fused(column,
                                    args,
                                    weight_collections,
                                    trainable,
                                    output_rank=2):
    # This option is only enabled for scattered_embedding_column.
    if args.hash_key:
        raise NotImplementedError("not implemented yet for hash_key")
    graph = ops.get_default_graph()
    partition_num = args.embedding_var_part_num
    if partition_num is None:
        partitioner = None
    else:
        partitioner = partitioned_variables.fixed_size_partitioner(
            partition_num)

    if args.shared_embedding_name is not None:
        raise NotImplementedError(
            "not implemented yet for shared_embedding_name")
    else:
        if args.use_embedding_var:
            embeddings = variable_scope.get_embedding_variable_internal(
                name="weights",
                embedding_dim=args.dimension,
                key_dtype=dtypes.int64,
                initializer=args.initializer,
                trainable=(trainable and args.trainable),
                collections=weight_collections,
                partitioner=partitioner,
                steps_to_live=args.steps_to_live,
                init_data_source=args.init_data_source,
                ht_partition_num=args.ht_partition_num,
                evconfig=args.evconfig)
            graph.add_to_collection(
                ops.GraphKeys.EMBEDDING_VARIABLES, embeddings)
        else:
            embeddings = contrib_variables.model_variable(
                name="weights",
                shape=[args.vocab_size, args.dimension],
                dtype=dtypes.float32,
                initializer=args.initializer,
                trainable=(trainable and args.trainable),
                collections=weight_collections)

    if fc._is_variable(embeddings):
        embeddings = [embeddings]
    else:
        embeddings = embeddings._get_variable_list()  # pylint: disable=protected-access
    # pylint: disable=protected-access
    fc._maybe_restore_from_checkpoint(column._checkpoint_path(), embeddings)
    return embedding_ops.safe_embedding_lookup_sparse(
        embeddings,
        args.input_tensor,
        sparse_weights=args.weight_tensor,
        combiner=args.combiner,
        name=column.name + "weights",
        max_norm=args.max_norm)


def input_from_feature_columns_fused(columns_to_tensors,
                                     feature_columns,
                                     weight_collections,
                                     trainable,
                                     scope,
                                     output_rank,
                                     default_name,
                                     cols_to_outs=None,
                                     blocknums=None):

    if weight_collections is not None:
        raise NotImplementedError("weight_collections not implemented yet.")
    if blocknums is not None:
        raise NotImplementedError("blocknums not implemented yet.")

    """Implementation of `input_from(_sequence)_feature_columns`."""
    columns_to_tensors = columns_to_tensors.copy()
    check_feature_columns(feature_columns)
    if cols_to_outs is not None and not isinstance(cols_to_outs, dict):
        raise ValueError('cols_to_outs must be a dict unless None')
    with variable_scope.variable_scope(scope,
                                       default_name=default_name,
                                       values=columns_to_tensors.values()):
        output_tensors = []
        transformer = _Transformer(columns_to_tensors)

        for column in sorted(set(feature_columns), key=lambda x: x.key):
            with variable_scope.variable_scope(None,
                                               default_name=column.name,
                                               values=columns_to_tensors.values()):
                transformed_tensor = transformer.transform(column)
                # pylint: disable=protected-access
                args = column._deep_embedding_lookup_arguments(
                    transformed_tensor)

                if cols_to_outs is not None:
                    cols_to_outs[column] = output_tensors[-1]
        return array_ops.concat(output_tensors, output_rank - 1)

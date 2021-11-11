from tensorflow.python.framework import ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import fused_embedding_ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers.feature_column_ops import check_feature_columns
from tensorflow.contrib.layers.python.layers.feature_column_ops import _Transformer
from tensorflow.contrib.layers.python.layers import feature_column as fc


def input_from_feature_columns_fused(columns_to_tensors,
                                     feature_columns,
                                     trainable=None,
                                     scope=None,
                                     cols_to_outs=None):
  """Implementation of `input_from(_sequence)_feature_columns`."""
  columns_to_tensors = columns_to_tensors.copy()
  check_feature_columns(feature_columns)
  if cols_to_outs is not None and not isinstance(cols_to_outs, dict):
    raise ValueError('cols_to_outs must be a dict unless None')
  with variable_scope.variable_scope(scope,
                                     default_name="input_from_feature_columns_fused",
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
        output = embeddings_from_arguments_fused(
            column, args, weight_collections, trainable)
        output_tensors.append(output[0])
        if cols_to_outs is not None:
          cols_to_outs[column] = output_tensors[-1]
  return array_ops.concat(output_tensors, 1)


def embeddings_from_arguments_fused(column,
                                    args,
                                    weight_collections,
                                    trainable):
    # This option is only enabled for scattered_embedding_column.
  if args.hash_key:
    raise NotImplementedError("not implemented yet for hash_key")

  graph = ops.get_default_graph()
  partition_num = args.embedding_var_part_num
  if partition_num is None:
    partitioner = None
  else:
    partitioner = partitioned_variables.fixed_size_partitioner(partition_num)

  # 1. get the embedding_weights
  if args.shared_embedding_name is not None:
    shared_embedding_collection_name = ("SHARED_EMBEDDING_COLLECTION_" +
                                        args.shared_embedding_name.upper())
    shared_embedding_collection = (
      graph.get_collection_ref(shared_embedding_collection_name))
    shape = [args.vocab_size, args.dimension]
    if shared_embedding_collection:
      if len(shared_embedding_collection) > 1:
        raise ValueError("Collection %s can only contain one "
                         "(partitioned) variable." %
                         shared_embedding_collection_name)
      else:
        embeddings = shared_embedding_collection[0]
        if (not args.use_embedding_var and embeddings.get_shape() != shape):
          raise ValueError("The embedding variable with name {} already "
                           "exists, but its shape does not match required "
                           "embedding shape here. Please make sure to use "
                           "different shared_embedding_name for different "
                           "shared embeddings.".format(args.shared_embedding_name))
    else:
      if args.use_embedding_var:
        embeddings = variable_scope.get_embedding_variable_internal(
          name=args.shared_embedding_name,
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
          name=args.shared_embedding_name,
          shape=shape,
          dtype=dtypes.float32,
          initializer=args.initializer,
          trainable=(trainable and args.trainable),
          collections=weight_collections)
      graph.add_to_collection(
        shared_embedding_collection_name, embeddings)
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

  # 2. look up

import argparse
import os
import sys
import tensorflow as tf

def get_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_checkpoint',
                      help='Full path of input checkpoint',
                      required=True)
  parser.add_argument('--output_checkpoint',
                      help='Full path of output checkpoint',
                      required=True)
  return parser

def is_belong_to_ev(name):
  ev_suffixes = ["-keys", "-values", "-versions", "-freqs",
                  "-keys_filtered", "-versions_filtered",
                  "-freqs_filtered", "-partition_offset",
                  "-partition_filter_offset"]
  for suffix in ev_suffixes:
    if suffix in name:
      return True
  return False

def get_part_id(name):
  part_str_start_index = name.rfind("/part_")
  part_id_start_index = part_str_start_index + 6
  for i in range(part_id_start_index, len(name)):
    if name[i] == "/":
      break
  if (i == len(name) - 1):
    return int(name[part_id_start_index: i + 1])
  else:
    return int(name[part_id_start_index: i])

def get_partition_number(name, variable_map):
  part_str_start_index = name.rfind("/part_")
  part_str_prefix = name[:part_str_start_index]
  for i in range(part_str_start_index + 1, len(name)):
    if name[i] == "/":
      break
  part_str_suffix = ""
  if i != len(name) - 1:
    part_str_suffix = name[i:]
  num_of_partition = 0
  while variable_map.__contains__(part_str_prefix+"/part_"
                                  + str(num_of_partition)
                                  + part_str_suffix+"-values"):
    num_of_partition += 1
  return num_of_partition

def is_primary_embedding_variable(name):
  part_str_start_index = name.rfind("/part_")
  for i in range(part_str_start_index + 1, len(name)):
    if name[i] == "/":
      return False
  return True

def build_partitioned_embedding_variable(name, shape, key_dtype,
                                         value_dtype, ev_option,
                                         ev_map, variable_map):
  num_of_partitions = get_partition_number(name, variable_map)
  partitioner = tf.fixed_size_partitioner(num_shards=num_of_partitions)
  var_name = name[: name.rfind("/part_")]
  with tf.device("/cpu:0"):
    var = tf.get_embedding_variable(var_name,
                                    embedding_dim=shape,
                                    key_dtype=key_dtype,
                                    value_dtype=value_dtype,
                                    partitioner=partitioner,
                                    ev_option=ev_option)
  ev_map[var_name] = var

def set_save_slice_info(primary_var, var):
  real_slot_name = var.name[var.name.rfind("/") + 1:-2]
  slice_info = primary_var._save_slice_info
  from tensorflow.python.ops import variables
  if isinstance(slice_info, variables.Variable.SaveSliceInfo):
    n = var.shape.ndims
    if n is None or n > 0:
      var._set_save_slice_info(variables.Variable.SaveSliceInfo(
          slice_info.full_name + "/" + real_slot_name,
          slice_info.full_shape[:n],
          slice_info.var_offset[:n],
          slice_info.var_shape[:n],
          var_full_name=slice_info.var_full_name + "/" +
          real_slot_name if slice_info.var_full_name else None))
    else:
      var._set_save_slice_info(
          slice_info.slot_save_slice_info(real_slot_name))

def main():
  reader = tf.train.load_checkpoint(args.input_checkpoint)
  shape_map = reader.get_variable_to_shape_map()
  dtype_map = reader.get_variable_to_dtype_map()
  names = sorted(shape_map.keys())
  ev_map = {}
  key_type_map = {}

  for name in names:
    if is_belong_to_ev(name):
      if "-keys" in name:
        var_name = name[:name.find("-keys")]
        key_type_map[var_name] = dtype_map[name]
      elif "-values" in name:
        var_name = name[:name.find("-value")]
        versions_name = var_name + "-versions"
        keys_name = var_name + "-keys"
        evict_option = None
        if shape_map[versions_name][0] != 0:
          evict_option = tf.GlobalStepEvict(steps_to_live=sys.maxsize)
        filter_option = tf.CounterFilter(filter_freq=sys.maxsize)
        ev_option = tf.EmbeddingVariableOption(
            filter_option=filter_option,
            evict_option=evict_option)
        if "/part_" in var_name:
          part_id = get_part_id(var_name)
          if is_primary_embedding_variable(var_name):
            if part_id == 0:
              build_partitioned_embedding_variable(var_name,
                                                   shape_map[name][1],
                                                   key_type_map[var_name],
                                                   dtype_map[name],
                                                   ev_option,
                                                   ev_map,
                                                   shape_map)
          else:
            with tf.device("/cpu:0"):
              var = tf.get_embedding_variable(var_name,
                                              embedding_dim=shape_map[name][1],
                                              key_dtype=key_type_map[var_name],
                                              value_dtype=dtype_map[name],
                                              ev_option=ev_option)
              primary_name = var_name[: var_name.rfind("/part_")]
              primary_var = list(ev_map[primary_name])[part_id]
              set_save_slice_info(primary_var, var)
        else:
          with tf.device("/cpu:0"):
            tf.get_embedding_variable(var_name,
                                      embedding_dim=shape_map[name][1],
                                      key_dtype=key_type_map[var_name],
                                      value_dtype=dtype_map[name],
                                      ev_option=ev_option)
    else:
      with tf.device("/cpu:0"):
        tf.get_variable(name, shape_map[name], dtype=dtype_map[name])

  saver = tf.train.Saver()
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    saver.restore(sess, args.input_checkpoint)
    saver.save(sess, args.output_checkpoint)

if __name__ == "__main__":
  parser = get_arg_parser()
  args = parser.parse_args()
  os.environ["TF_EV_SAVE_FILTERED_FEATURES"] = "false"
  main()
  del os.environ["TF_EV_SAVE_FILTERED_FEATURES"]

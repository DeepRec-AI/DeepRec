## Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.python.lib.io import file_io

FLAGS = None


def _write_assets(assets_directory, assets_filename):
  """Writes asset files to be used with SavedModel for half plus two.

  Args:
    assets_directory: The directory to which the assets should be written.
    assets_filename: Name of the file to which the asset contents should be
        written.

  Returns:
    The path to which the assets file was written.
  """
  if not file_io.file_exists(assets_directory):
    file_io.recursive_create_dir(assets_directory)

  path = os.path.join(
      tf.compat.as_bytes(assets_directory), tf.compat.as_bytes(assets_filename))
  file_io.write_string_to_file(path, "asset-file-contents")
  return path


def _build_regression_signature(input_tensor, output_tensor):
  """Helper function for building a regression SignatureDef."""
  input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
  signature_inputs = {
      tf.saved_model.signature_constants.REGRESS_INPUTS: input_tensor_info
  }
  output_tensor_info = tf.saved_model.utils.build_tensor_info(output_tensor)
  signature_outputs = {
      tf.saved_model.signature_constants.REGRESS_OUTPUTS: output_tensor_info
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.REGRESS_METHOD_NAME)


# Possibly extend this to allow passing in 'classes', but for now this is
# sufficient for testing purposes.
def _build_classification_signature(input_tensor, scores_tensor):
  """Helper function for building a classification SignatureDef."""
  input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
  signature_inputs = {
      tf.saved_model.signature_constants.CLASSIFY_INPUTS: input_tensor_info
  }
  output_tensor_info = tf.saved_model.utils.build_tensor_info(scores_tensor)
  signature_outputs = {
      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
          output_tensor_info
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)


def _generate_saved_model(export_dir,
                          checkpoint_dir,
                          as_text=False,
                          use_main_op=False,
                          device_type="cpu"):
  """Generates SavedModel for half plus two.

  Args:
    export_dir: The directory to which the SavedModel should be written.
    as_text: Writes the SavedModel protocol buffer in text format to disk.
    use_main_op: Whether to supply a main op during SavedModel build time.
    device_name: Device to force ops to run on.
  """
  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

  device_name = "/cpu:0"
  if device_type == "gpu":
    device_name = "/gpu:0"

  with tf.Session(
      graph=tf.Graph(),
      config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.device(device_name):
      # Set up the model parameters as variables to exercise variable loading
      # functionality upon restore.
      a = tf.Variable(0.5, name="a")
      b = tf.Variable(2.0, name="b")
      c = tf.Variable(3.0, name="c")

      # Create a placeholder for serialized tensorflow.Example messages to be
      # fed.
      serialized_tf_example = tf.placeholder(tf.string, name="tf_example")

      # Parse the tensorflow.Example looking for a feature named "x" with a
      # single floating point value.
      feature_configs = {
          "x": tf.FixedLenFeature([1], dtype=tf.float32),
          "x2": tf.FixedLenFeature([1], dtype=tf.float32, default_value=[0.0])
      }
      # parse_example only works on CPU
      with tf.device("/cpu:0"):
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
      # Use tf.identity() to assign name
      x = tf.identity(tf_example["x"], name="x")
      y = tf.add(tf.multiply(a, x), b, name="y")
      y2 = tf.add(tf.multiply(a, x), c, name="y2")

      x2 = tf.identity(tf_example["x2"], name="x2")
      y3 = tf.add(tf.multiply(a, x2), c, name="y3")

    # Create an assets file that can be saved and restored as part of the
    # SavedModel.
    original_assets_directory = "/tmp/original/export/assets"
    original_assets_filename = "foo.txt"
    original_assets_filepath = _write_assets(original_assets_directory,
                                             original_assets_filename)

    # Set up the assets collection.
    assets_filepath = tf.constant(original_assets_filepath)
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, assets_filepath)
    filename_tensor = tf.Variable(
        original_assets_filename,
        name="filename_tensor",
        trainable=False,
        collections=[])
    assign_filename_op = filename_tensor.assign(original_assets_filename)

    # Set up the signature for Predict with input and output tensor
    # specification.
    predict_input_tensor = tf.saved_model.utils.build_tensor_info(x)
    predict_signature_inputs = {"x": predict_input_tensor}

    predict_output_tensor = tf.saved_model.utils.build_tensor_info(y)
    predict_signature_outputs = {"y": predict_output_tensor}
    predict_signature_def = (
        tf.saved_model.signature_def_utils.build_signature_def(
            predict_signature_inputs, predict_signature_outputs,
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    signature_def_map = {
        "regress_x_to_y":
            _build_regression_signature(serialized_tf_example, y),
        "regress_x_to_y2":
            _build_regression_signature(serialized_tf_example, y2),
        "regress_x2_to_y3":
            _build_regression_signature(x2, y3),
        "classify_x_to_y":
            _build_classification_signature(serialized_tf_example, y),
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            predict_signature_def
    }
    # Initialize all variables and then save the SavedModel.
    sess.run(tf.global_variables_initializer())

    if use_main_op:
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map=signature_def_map,
          assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
          main_op=tf.group(tf.saved_model.main_op.main_op(),
                           assign_filename_op))
    else:
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map=signature_def_map,
          assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
          legacy_init_op=tf.group(assign_filename_op))

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    checkpoint_path = saver.save(sess,
                                 save_path=os.path.join(checkpoint_dir,
                                                       'DeepFM-checkpoint'),
                                 global_step=1)

  builder.save(as_text)


def main(_):
  _generate_saved_model(
      FLAGS.saved_model_dir, FLAGS.checkpoint_dir)
  print("SavedModel generated at: %(saved_model_dir)s, %(checkpoint_dir)s" % {
      "saved_model_dir": FLAGS.saved_model_dir,
      "checkpoint_dir": FLAGS.checkpoint_dir
  })

  '''
  _generate_saved_model(
      FLAGS.saved_model_dir_pbtxt, FLAGS.checkpoint_dir, as_text=True)
  print("SavedModel generated at: %(saved_model_dir)s, %(checkpoint_dir)s" % {
      "saved_model_dir": FLAGS.saved_model_dir_pbtxt,
      "checkpoint_dir": FLAGS.checkpoint_dir
  })

  _generate_saved_model(
      FLAGS.saved_model_dir_main_op, FLAGS.checkpoint_dir, use_main_op=True)
  print("SavedModel generated at: %(saved_model_dir)s, %(checkpoint_dir)s" % {
      "saved_model_dir": FLAGS.saved_model_dir_main_op,
      "checkpoint_dir": FLAGS.checkpoint_dir
  })
  '''

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--saved_model_dir",
      type=str,
      default="/tmp/saved_model",
      help="Directory where to output SavedModel.")
  parser.add_argument(
      "--saved_model_dir_pbtxt",
      type=str,
      default="/tmp/saved_model_pbtxt",
      help="Directory where to output the text format of SavedModel.")
  parser.add_argument(
      "--saved_model_dir_main_op",
      type=str,
      default="/tmp/saved_model_main_op",
      help="Directory where to output the SavedModel with a main op.")
  parser.add_argument(
      "--checkpoint_dir",
      type=str,
      default="/tmp/checkpoint/1",
      help="Directory where to output Checkpoint.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

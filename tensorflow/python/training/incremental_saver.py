# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=invalid-name
"""Save and restore variables."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import re
import threading
import time
import uuid
import enum

import numpy as np
import six

from google.protobuf import text_format
from multiprocessing.pool import ThreadPool

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training.saver import BaseSaverBuilder 
from tensorflow.python.training.saver import Saver 
from tensorflow.python.training.saver import get_checkpoint_mtimes
from tensorflow.python.training.saver import get_checkpoint_state
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


SUB_INCR_CKPT_DIR = ".incremental_checkpoint"
class BuildIncrMode(enum.Enum):
  NOTHING = 0
  ACTIVATE_OPS = 1
  INCR_SAVE_RESTORE_OPS = 2


class IncrementalSaverBuilder(BaseSaverBuilder):
   
  def incremental_save_op(self, filename_tensor, saveables):
    tensor_names = []
    tensor_slices = []
    tensors = []
    is_sparses = []
    for saveable in saveables:
      if isinstance(saveable, BaseSaverBuilder.EmbeddingVariableSaveable):
        tensors.append(saveable.handle_op)
        tensor_slices.append("")
        tensor_names.append(saveable.name)
        is_sparses.append(True)
      else:
        for spec in saveable.specs:
          tensor_name, is_sparse = self._GetTensorNameAndIsSparse(spec, saveable)
          tensors.append(spec.tensor)
          tensor_slices.append(spec.slice_spec)
          tensor_names.append(tensor_name)
          is_sparses.append(is_sparse)
    return io_ops.incr_save(filename_tensor, tensor_names, tensor_slices, is_sparses, tensors)


  # pylint: enable=unused-argument
  def incremental_restore_op(self, filename_tensor, saveable, preferred_shard):
    tensors = []
    original_tensors = self.restore_op(
        self.filename_tensor, saveable, preferred_shard)
    
    if isinstance(saveable, BaseSaverBuilder.EmbeddingVariableSaveable):
      spec = saveable.specs[0]
      tensors = gen_io_ops.incr_restore(filename_tensor,
                                        [spec.name + "-keys", spec.name + "-values", spec.name + "-versions"],
                                        [spec.slice_spec],
                                        [True],
                                        original_tensors)
    else:
      for spec in saveable.specs:
        tensor_name, is_sparse = self._GetTensorNameAndIsSparse(spec, saveable)
        tensors.append(
          io_ops.incr_restore(
              filename_tensor,
              [tensor_name],
              [spec.slice_spec],
              [is_sparse],
              original_tensors)[0])
    return tensors


  def _AddSaveOps(self, filename_tensor, saveables):
    """ Override BaseSaverBuilder._AddSaveOps
    Add ops to save variables that are on the same shard.

    Args:
      filename_tensor: String Tensor.
      saveables: A list of SaveableObject objects.

    Returns:
      A tensor with the filename used to save.
    """
    increment_save = self.incremental_save_op(filename_tensor, saveables)
    return control_flow_ops.with_dependencies([increment_save], filename_tensor)

  def _AddRestoreOps(self,
                     filename_tensor,
                     saveables,
                     restore_sequentially,
                     reshape,
                     restore_rules={},
                     preferred_shard=-1,
                     name="restore_all"):
    """Add operations to restore saveables.

    Args:
      filename_tensor: Tensor for the path of the file to load.
      saveables: A list of SaveableObject objects.
      restore_sequentially: True if we want to restore variables sequentially
        within a shard.
      reshape: True if we want to reshape loaded tensors to the shape of
        the corresponding variable.
      preferred_shard: Shard to open first when loading a sharded file.
      name: Name for the returned op.

    Returns:
      An Operation that restores the variables.
    """
    #all_tensors = self.bulk_restore(filename_tensor, saveables, preferred_shard,
    #                                restore_sequentially)

    assign_ops = []
    idx = 0
    # Load and optionally reshape on the CPU, as string tensors are not
    # available on the GPU.
    # TODO(touts): Re-enable restore on GPU when we can support annotating
    # string tensors as "HostMemory" inputs.
    for saveable in saveables:
      restore_control_inputs = assign_ops[-1:] if restore_sequentially else []
      # Load and optionally reshape on the CPU, as string tensors are not
      # available on the GPU.
      # TODO(touts): Re-enable restore on GPU when we can support annotating
      # string tensors as "HostMemory" inputs.
      with ops.device(saveable_object_util.set_cpu0(saveable.device) if saveable.device else None):
        with ops.control_dependencies(restore_control_inputs):
          if saveable.name in restore_rules and \
              not restore_rules[saveable.name]:
            if not hasattr(saveable.op, 'initializer'):
              raise ValueError(
                  "Saveable %s should have attribute `initializer`" % \
                      saveable.name)
            tensors = saveable.op.initializer.outputs
            logging.warning(
                "Saveable %s not found and will be reinitialized.",
                saveable.name)
          else :
            tensors = self.incremental_restore_op(
                filename_tensor, saveable, preferred_shard)
          shapes = None
          if reshape:
            # Compute the shapes, let the restore op decide if and how to do
            # the reshape.
            shapes = []
            for spec in saveable.specs:
              v = spec.tensor
              shape = v.get_shape()
              if not shape.is_fully_defined():
                shape = array_ops.shape(v)
              shapes.append(shape)
          if saveable.name in restore_rules and restore_rules[saveable.name]:
            tensors = [restore_rules[saveable.name](t, i, shapes) \
                for i, t in enumerate(tensors)]
            logging.warning(
                "Saveable %s found and will be transformed.", saveable.name)
          if isinstance(saveable, BaseSaverBuilder.EmbeddingVariableSaveable):
            assign_ops.append(saveable.incr_restore([filename_tensor], shapes))
          else:
            assign_ops.append(saveable.restore(tensors, shapes))

    # Create a Noop that has control dependencies from all the updates.
    return control_flow_ops.group(*assign_ops, name=name)

  def _build_internal(self,
                      names_to_saveables,
                      reshape=False,
                      sharded=False,
                      max_to_keep=5,
                      keep_checkpoint_every_n_hours=10000.0,
                      name=None,
                      restore_sequentially=False,
                      filename="model",
                      build_save=True,
                      build_restore=True):
    if context.executing_eagerly(): 
      raise ValueError("IncrementalSaver not support in eager mode")

    saveables = saveable_object_util.validate_and_slice_inputs(
        names_to_saveables)
    if max_to_keep is None:
      max_to_keep = 0

    if build_restore:
      saveable_names = set()
      if isinstance(names_to_saveables, dict):
        saveable_names = set(names_to_saveables.keys())
      else:
        saveable_names = set([s.name for s in names_to_saveables])

      checkpoint_saveable_names = set()
      if filename is not None:
        try:
          checkpoint_reader = pywrap_tensorflow.NewCheckpointReader(filename)
          checkpoint_saveable_names = set(
              checkpoint_reader.get_variable_to_shape_map().keys())
        except:
          pass

      unready_saveable_names = saveable_names - checkpoint_saveable_names
      '''
      initializables = set([s for s, r in six.iteritems(Saver._RESTORE_RULES) \
          if s in unready_saveable_names and not r])
      restore_rules = {s:r for s, r in six.iteritems(Saver._RESTORE_RULES) \
          if s in checkpoint_saveable_names and r}
      restore_rules.update({i: False for i in initializables})
      logging.info('Build IncrSaver with restore rules: %s', restore_rules)
      '''

    with ops.name_scope(name, "incr_save",
                        [saveable.op for saveable in saveables]) as name:
      # Add the Constant string tensor for the filename.
      #filename_tensor = constant_op.constant(filename or "model")
      self._incremental_filename_tensor = constant_op.constant(filename or "model")
      self._incremental_save_tensor = None
      self._incremental_restore_tensor = None

      # Add the save ops.
      if sharded:
        per_device = self._GroupByDevices(saveables)
        self._incremental_save_tensor = self._AddShardedSaveOps(
              self._incremental_filename_tensor,
              per_device)
        self._incremental_restore_tensor = self._AddShardedRestoreOps(
              self._incremental_filename_tensor, per_device,
              restore_sequentially, reshape)
      else:
        self._incremental_save_tensor = self._AddSaveOps(
              self._incremental_filename_tensor,
              saveables)
        self._incremental_restore_tensor = self._AddRestoreOps(
              self._incremental_filename_tensor, saveables,
              restore_sequentially, reshape)

    return saver_pb2.SaverDef(
        filename_tensor_name=self._incremental_filename_tensor.name,
        save_tensor_name=self._incremental_save_tensor.name,
        restore_op_name=self._incremental_restore_tensor.name,
        max_to_keep=max_to_keep,
        sharded=sharded,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        version=self._write_version)

def _get_incremental_saver(incremental_save_restore, full_saver):
  if incremental_save_restore:
    incr_saver = IncrementalSaver(sharded=True, allow_empty=True, saver_def=full_saver.saver_def, defer_build=True,
                                  incremental_include_normal_var=full_saver._incremental_include_normal_var)
    incr_saver.build(full_saver._builder.filename_tensor)
    return incr_saver
  else:
    return None


class IncrementalSaver(Saver):

  def build(self, filename_tensor):
    if context.executing_eagerly():
      raise RuntimeError("Use save/restore instead of build in eager mode.")
    self.incr_saver_def = None
    self.filename_tensor = filename_tensor
    self._build(self._filename, build_save=True, build_restore=True)


  def _build(self, checkpoint_path, build_save, build_restore):
    self._last_incr_checkpoints = []
    self._incr_checkpoints_to_be_deleted = []

    if not context.executing_eagerly():
      if self._is_built:
        return
      self._is_built = True

    if self._builder is None:
      self._builder = IncrementalSaverBuilder(self._write_version, build_incr_activateop=True,
                                              incremental_include_normal_var=self._incremental_include_normal_var)
    
    self._builder.filename_tensor = self.filename_tensor
    if self._var_list is None:
      # pylint: disable=protected-access
      self._var_list = variables._all_saveable_objects()
      if ops.get_collection('vars@loss_scaling'):
        loss_scale_vars = ops.get_collection('vars@loss_scaling')
        self._var_list = list(set(self._var_list) -  set(loss_scale_vars))
    if not self._var_list:
      if self._allow_empty:
        self._is_empty = True
        return
      else:
        raise ValueError("No variables to save")
    self._is_empty = False

    self.incr_saver_def = self._builder._build_internal(  # pylint: disable=protected-access
          self._var_list,
          reshape=self._reshape,
          sharded=self._sharded,
          max_to_keep=self._max_to_keep,
          keep_checkpoint_every_n_hours=self._keep_checkpoint_every_n_hours,
          name=self._name,
          restore_sequentially=self._restore_sequentially,
          filename=checkpoint_path,
          build_save=build_save, build_restore=build_restore)
    if not context.executing_eagerly():
      # Updates next checkpoint time.
      # Set in __init__ when executing eagerly.
      self._next_checkpoint_time = (
          time.time() + self._keep_checkpoint_every_n_hours * 3600)


  def _RecordLastIncrCheckpoint(self, latest_save_path):
    if not self.saver_def.max_to_keep:
      return
    # Remove first from list if the same name was used before.
    for p in self._last_incr_checkpoints:
      if latest_save_path == self._CheckpointFilename(p):
        self._last_incr_checkpoints.remove(p)
    # Append new path to list
    self._last_incr_checkpoints.append((latest_save_path, time.time()))

    # For incremental_save only save last 2 ckpt
    if len(self._last_incr_checkpoints) > self.incr_saver_def.max_to_keep:
      self._incr_checkpoints_to_be_deleted.append(self._last_incr_checkpoints.pop(0))

  def _MaybeDeleteOldIncrCheckpoints(self, meta_graph_suffix="meta"):
    if self._incr_checkpoints_to_be_deleted:
      p = self._incr_checkpoints_to_be_deleted.pop(0)
      # Do not delete the file if we keep_checkpoint_every_n_hours is set and we
      # have reached N hours of training.
      #should_keep = p[1] > self._next_checkpoint_time
      #if should_keep:
      #  self._next_checkpoint_time += (
      #      self.saver_def.keep_checkpoint_every_n_hours * 3600)
      #  return

      # Otherwise delete the files.
      try:
        checkpoint_prefix = self._CheckpointFilename(p)
        #checkpoint_management._delete_file_if_exists(
        #    self._MetaGraphFilename(checkpoint_prefix, meta_graph_suffix))
        if self.saver_def.version == saver_pb2.SaverDef.V2:
          # V2 has a metadata file and some data files.
          checkpoint_management._delete_file_if_exists(checkpoint_prefix + ".index")
          checkpoint_management._delete_file_if_exists(checkpoint_prefix +
                                      ".data-?????-of-?????")
        else:
          # V1, Legacy.  Exact match on the data file.
          checkpoint_management._delete_file_if_exists(checkpoint_prefix)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("Ignoring: %s", str(e))

  @property
  def last_incr_checkpoints(self):
    return list(self._CheckpointFilename(p) for p in self._last_incr_checkpoints)

  def set_last_incr_checkpoints(self, last_checkpoints):
    assert isinstance(last_checkpoints, list)
    self._last_incr_checkpoints = [(s, np.inf) for s in last_checkpoints]

  def set_last_incr_checkpoints_with_time(self, last_checkpoints_with_time):
    assert isinstance(last_checkpoints_with_time, list)
    self._last_incr_checkpoints = last_checkpoints_with_time

  def recover_last_incr_checkpoints(self, checkpoint_paths):
    mtimes = get_checkpoint_mtimes(checkpoint_paths)
    self.set_last_incr_checkpoints_with_time(list(zip(checkpoint_paths, mtimes)))


  def incremental_save(self,
          sess,
          save_path,
          global_step=None,
          meta_graph_suffix="meta",
          write_state=True):
    if context.executing_eagerly():
      logging.warning("`incremental_save()` not support in Eager mode")
      return
    if not self._is_built and not context.executing_eagerly():
      raise RuntimeError(
          "`build()` should be called before save if defer_build==True")
    latest_filename = "checkpoint"
    if self._write_version != saver_pb2.SaverDef.V2:
      logging.warning("*******************************************************")
      logging.warning("TensorFlow's V1 checkpoint format has been deprecated.")
      logging.warning("Consider switching to the more efficient V2 format:")
      logging.warning("   `tf.train.Saver(write_version=tf.train.SaverDef.V2)`")
      logging.warning("now on by default.")
      logging.warning("*******************************************************")

    if global_step is not None:
      if not isinstance(global_step, compat.integral_types):
        global_step = training_util.global_step(sess, global_step)
      checkpoint_file = "%s-%d" % (save_path, global_step)
      if self._pad_step_number:
        # Zero-pads the step numbers, so that they are sorted when listed.
        checkpoint_file = "%s-%s" % (save_path, "{:08d}".format(global_step))
    else:
      checkpoint_file = save_path
      if os.path.basename(
          save_path) == latest_filename and not self._sharded:
        # Guard against collision between data file and checkpoint state file.
        raise ValueError(
            "'latest_filename' collides with 'save_path': '%s' and '%s'" %
            (latest_filename, save_path))

    if not isinstance(sess, session.SessionInterface):
      raise TypeError("'sess' must be a Session; %s" % sess)

    if not gfile.IsDirectory(os.path.dirname(save_path)):
      gfile.MakeDirs(os.path.dirname(save_path))
    save_path_parent = os.path.dirname(save_path)
    if not self._is_empty:
      try:
        model_checkpoint_path = sess.run(
            self._builder._incremental_save_tensor.name,
            {self._builder._incremental_filename_tensor.name: checkpoint_file})

        model_checkpoint_path = compat.as_str(model_checkpoint_path)
        if write_state:
          self._RecordLastIncrCheckpoint(model_checkpoint_path)
          checkpoint_management.update_checkpoint_state_internal(
              save_dir=save_path_parent,
              model_checkpoint_path=model_checkpoint_path,
              all_model_checkpoint_paths=self.last_incr_checkpoints,
              latest_filename=latest_filename,
              save_relative_paths=self._save_relative_paths)
          self._MaybeDeleteOldIncrCheckpoints(meta_graph_suffix=meta_graph_suffix)
      except (errors.FailedPreconditionError, errors.NotFoundError) as exc:
        if not gfile.IsDirectory(save_path_parent):
          exc = ValueError(
              "Parent directory of {} doesn't exist, can't save.".format(
                  save_path))
        raise exc

    if self._is_empty:
      return None
    else:
      return model_checkpoint_path

  def incremental_restore(self, sess, save_path, incr_save_path):
    if context.executing_eagerly():
      logging.warning("`incremental_restore()` not support in Eager mode")
      return
    if self._is_empty:
      return
    if save_path is None:
      raise ValueError("Can't load save_path when it is None.")
    logging.info("Incremental restoring parameters from %s", incr_save_path)
    sess.run(self._builder._incremental_restore_tensor.name,
             {self.saver_def.filename_tensor_name: save_path,
             self._builder._incremental_filename_tensor.name: incr_save_path})


  def recover_incr_checkpoints(self, sess, checkpoint_dir):
    incr_checkpoint_dir = os.path.join(checkpoint_dir, ".incremental_checkpoint")
    incr_ckpt = get_checkpoint_state(incr_checkpoint_dir)
    ckpt = get_checkpoint_state(checkpoint_dir)

    def get_ckpt_global_step(ckpt):
      return ckpt[ckpt.rfind('-') + 1:]

    need_incremental_restore = False
    need_recover_last_ckpt = False
    if incr_ckpt:
      try:
        ckpt_gloabl_step = int(get_ckpt_global_step(ckpt.model_checkpoint_path))
        incr_ckpt_gloabl_step = int(get_ckpt_global_step(incr_ckpt.model_checkpoint_path))
        if ckpt_gloabl_step < incr_ckpt_gloabl_step:
          need_incremental_restore = True
        else:
          logging.info("incremental checkpoint is older than checkpoint, skip incremental_restore()")
        need_recover_last_ckpt = True
      except Exception:
        logging.info("checkpoint or incremental checkpoint doesn't have gloabl_step info, "
                "checkpoint is [%s], incremental checkpoint is [%s]."
                % (ckpt.model_checkpoint_path, incr_ckpt.model_checkpoint_path))
    if need_incremental_restore:
      self.incremental_restore(sess, ckpt.model_checkpoint_path, incr_ckpt.model_checkpoint_path)
    if need_recover_last_ckpt:
      self.recover_last_incr_checkpoints(incr_ckpt.all_model_checkpoint_paths)

  def export_meta_graph(self,
                        filename=None,
                        collection_list=None,
                        as_text=False,
                        export_scope=None,
                        clear_devices=False,
                        clear_extraneous_savers=False,
                        strip_default_attrs=False,
                        save_debug_info=False):
    return saver_module.export_meta_graph(
        filename=filename,
        graph_def=ops.get_default_graph().as_graph_def(add_shapes=True),
        saver_def=self.saver_def,
        collection_list=collection_list,
        as_text=as_text,
        export_scope=export_scope,
        clear_devices=clear_devices,
        clear_extraneous_savers=clear_extraneous_savers,
        strip_default_attrs=strip_default_attrs,
        save_debug_info=save_debug_info,
        incr_saver_def=self.incr_saver_def)


from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training.tracking import tracking

class GraphState(object):
  ORIGINAL = 0
  CALIBRATE = 1
  INFERENCE = 2

def CreateConverter(run_params, saved_model_dir, session_config,
                    conversion_params):
  """Return a TrtGraphConverter."""
  return trt_convert.TrtGraphConverter(
      input_saved_model_dir=saved_model_dir,
      session_config=session_config,
      max_batch_size=conversion_params.max_batch_size,
      max_workspace_size_bytes=conversion_params.max_workspace_size_bytes,
      precision_mode=conversion_params.precision_mode,
      minimum_segment_size=conversion_params.minimum_segment_size,
      is_dynamic_op=conversion_params.is_dynamic_op,
      maximum_cached_engines=conversion_params.maximum_cached_engines,
      use_calibration=conversion_params.use_calibration,
      use_ev=conversion_params.use_ev)

def GetGPUOptions():
  gpu_options = config_pb2.GPUOptions()
  gpu_options.allow_growth = True
  return gpu_options

def GetConfigProto(run_params, conversion_params, graph_state):
  """Get config proto based on specific settings."""
  if graph_state == GraphState.INFERENCE and run_params['convert_online']:
    rewriter_cfg = trt_convert.get_tensorrt_rewriter_config(conversion_params)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_cfg)
  else:
    graph_options = config_pb2.GraphOptions()
    if conversion_params.rewriter_config_template is not None:
      graph_options.rewrite_options.CopyFrom(
          conversion_params.rewriter_config_template)

  config = config_pb2.ConfigProto(
      gpu_options=GetGPUOptions(), graph_options=graph_options)
  return config

def GetConversionParams(run_params):
  """Return a TrtConversionParams for test."""
  return trt_convert.TrtConversionParams(
      # We use the minimum of all the batch sizes, so when multiple different
      # input shapes are provided it'll always create new engines in the
      # cache, and we can therefore test the cache behavior.
      rewriter_config_template=None,
      max_workspace_size_bytes=1 << 25, 
      precision_mode=run_params['precision_mode'],
      minimum_segment_size=run_params['minimum_segment_size'],
      is_dynamic_op=run_params['dynamic_engine'],
      maximum_cached_engines=1,
      use_calibration=run_params['use_calibration'],
      max_batch_size=run_params['max_batch_size'],
      use_ev=run_params['use_ev'])

def ConvertGraph(run_params, saved_model_dir, trt_saved_model_dir):
  """Return trt converted graphdef."""
  conversion_params = GetConversionParams(run_params)
  logging.info(conversion_params)

  session_config = GetConfigProto(
      run_params, conversion_params, GraphState.INFERENCE)
  logging.info("Creating TRT graph for inference, config\n%s",
               str(session_config))

  converter = CreateConverter(run_params, saved_model_dir,
                              session_config, conversion_params)
  converter.convert()

  converter.save(trt_saved_model_dir)

  return trt_saved_model_dir


if __name__ == "__main__":
  run_params = {
    'precision_mode':"FP32",
    'dynamic_engine': True,
    'use_calibration': False,
    'max_batch_size': 1024,
    'convert_online': False,
    'minimum_segment_size': 4,
    'use_ev': True,
  }

  saved_model_dir = '/model/pb'
  trt_saved_model_dir = './trtmodel'
  ConvertGraph(run_params, saved_model_dir, trt_saved_model_dir)


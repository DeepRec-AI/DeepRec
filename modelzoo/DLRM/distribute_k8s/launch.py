#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os 
import sys
import subprocess as sp

DEFAULT_SEASTAR_PORT="3333"
JEMALLOC_244 = "libjemalloc.so.2.4.4"
JEMALLOC_251 = "libjemalloc.so.2.5.1"

def gen_cluster_info(workspace):
  tf_config_json = os.environ.get("TF_CONFIG", "{}")
  print("TF_CONFIG=", tf_config_json)
  tf_config = json.loads(tf_config_json)
  cluster = tf_config.get("cluster", {})
  if cluster is None:
    print("TF_CONFIG cluster is empty")
    return

  ps_hosts = []
  worker_hosts = []
  chief_hosts = []
  node_list = []
  for key, value in cluster.items():
    if "ps" == key:
      ps_hosts = value
    elif "worker" == key:
      worker_hosts = value
    elif "chief" == key:
      chief_hosts = value
    node_list.extend(value)

  os.environ['TF_SEASTAR_ENDPOINT_MAP_PATH'] = '/tmp/'
  print("Start to gen endpoint_map file.")
  #endpoint_map_path = os.path.join(workspace, ".endpoint_map")
  endpoint_map_path = "/tmp/.endpoint_map"
  with open(endpoint_map_path, 'w') as fout:
   for node in node_list:
     host = node[0:node.index(':')]
     fout.write(node + "=" + host + ":" + DEFAULT_SEASTAR_PORT + "\n")
  os.system("ls -ltr /tmp/.endpoint_map")

  task = tf_config.get("task", {})
  if task is None:
    print("TF_CONFIG task is empty")
    return

  task_index = task['index']
  job_name = task['type']
  return ps_hosts, worker_hosts, chief_hosts, job_name, task_index

def copy_python_binary(local_dir):
  cmd_str = "cp /usr/bin/python " + os.path.join(local_dir, "python_bin")
  return sp.call(cmd_str, shell=True)

def set_jemalloc_version(workspace):
  strategy = os.environ.get("MEM_USAGE_STRATEGY", "")
  cmd_str = ""
  if "xmin" == strategy:
    cmd_str = "export JEMALLOC_VERSION=" + os.path.join(workspace, JEMALLOC_244) + ";"
    cmd_str += "export MALLOC_CONF=decay_time:0;"
  elif "xmid" == strategy:
    cmd_str = "export JEMALLOC_VERSION=" + os.path.join(workspace, JEMALLOC_244) + ";"
  elif "min" == strategy:
    cmd_str = "export JEMALLOC_VERSION=" + os.path.join(workspace, JEMALLOC_251) + ";"
    cmd_str += "export MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0;"
  elif "mid" == strategy:
    cmd_str = "export JEMALLOC_VERSION=" + os.path.join(workspace, JEMALLOC_251) + ";"
    cmd_str += "export MALLOC_CONF=background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:10000;"
  elif "max" == strategy:
    cmd_str = "export JEMALLOC_VERSION=" + os.path.join(workspace, JEMALLOC_251) + ";"
    cmd_str += "export MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:240000,muzzy_decay_ms:240000;"
  elif "244" == strategy:
    cmd_str = "export JEMALLOC_VERSION=" + os.path.join(workspace, JEMALLOC_244) + ";"
  elif "251" == strategy:
    cmd_str = "export JEMALLOC_VERSION=" + os.path.join(workspace, JEMALLOC_251) + ";"
    cmd_str += "export MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:60000,muzzy_decay_ms:60000;"
  elif "close" == strategy:
    pass
  else:
    cmd_str = "export JEMALLOC_VERSION=" + os.path.join(workspace, JEMALLOC_251) + ";"
    cmd_str += "export MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:240000,muzzy_decay_ms:240000;"
  
  return cmd_str

def pip_install_requirements(workspace):
  requirements_path = os.path.join(workspace, "requirements.txt")
  if not os.path.exists(requirements_path):
    return 0
  
  cmd_str = "$(which pip) install -r " + requirements_path
  print("try to install requirements.txt from " + requirements_path)
  return sp.call(cmd_str, shell=True)

def run_tensorflow_job(workspace, tf_script, tf_args, tf_envs, set_jemalloc_version_cmd):
  cmd_str = "cd " + workspace + ";"
  if set_jemalloc_version_cmd:
    cmd_str += set_jemalloc_version_cmd 
    cmd_str += "LD_PRELOAD=${JEMALLOC_VERSION} " 
  cmd_str += " ".join(tf_envs) + " $(which python) -u "
  cmd_str += tf_script + " " + " ".join(tf_args)
  print("run tensorflow command:", cmd_str)
  return sp.call(cmd_str, shell=True)

def set_mkl_envs(job_name):
  envs = []
  if "ps" == job_name:
    envs.append("OMP_NUM_THREADS=1")
    envs.append("KMP_BLOCKTIME=0")
    envs.append("MKL_ENABLE_INSTRUCTIONS=AVX2")
  elif "worker" == job_name:
    envs.append("OMP_NUM_THREADS=6")
    envs.append("KMP_BLOCKTIME=0")
    envs.append("MKL_ENABLE_INSTRUCTIONS=AVX2")
  elif "evaluator" == job_name or "chief" == job_name:
    envs.append("OMP_NUM_THREADS=1")
    envs.append("KMP_BLOCKTIME=0")
    envs.append("MKL_ENABLE_INSTRUCTIONS=AVX2")
  else:
    envs.append("OMP_NUM_THREADS=1")
    envs.append("KMP_BLOCKTIME=0")
    envs.append("MKL_ENABLE_INSTRUCTIONS=AVX2")

  return envs

def set_network_threads(job_name):
  envs = []
  if "ps" == job_name:
    envs.append("WORKER_DEFAULT_CORE_NUM=24")
  elif "worker" == job_name:
    envs.append("PS_DEFAULT_CORE_NUM=24")
  return envs

if __name__ == "__main__":
  print("start launching tensorflow job")

  if "TF_WORKSPACE" not in os.environ:
    print("TF_WORKSPACE env should be set.")
    exit(1)
  workspace = os.environ.get("TF_WORKSPACE", "")

  if "TF_SCRIPT" not in os.environ:
    print("TF_SCRIPT env should be set.")
    exit(1)
  tf_script = os.environ.get("TF_SCRIPT", "")

  if "JEMALLOC_PATH" not in os.environ:
    jemalloc_path = workspace
  else:
    jemalloc_path = os.environ.get("JEMALLOC_PATH", "")


  #ret_code = copy_python_binary(workspace)
  #if (ret_code != 0):
  #  exit(ret_code)

  tf_args = sys.argv[1:] 

  tf_envs = []
  #tf_envs.append("TF_SEASTAR_ENDPOINT_MAP_PATH=/tmp/")
  if "TF_CONFIG" in os.environ:
    ps_hosts, worker_hosts, chief_hosts, job_name, task_index = gen_cluster_info(workspace)
    
    os.environ["TASK_INDEX"] = str(task_index)
    os.environ["JOB_NAME"] = str(job_name)
    #tf_envs.extend(set_mkl_envs(job_name))

  set_jemalloc_version_cmd = set_jemalloc_version(jemalloc_path)
  
  ret_code = pip_install_requirements(workspace)
  if (ret_code != 0):
    exit(ret_code)

  ret_code = run_tensorflow_job(workspace, tf_script, tf_args, tf_envs, set_jemalloc_version_cmd)
  if (ret_code != 0):
    exit(ret_code)

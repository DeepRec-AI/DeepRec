#!/usr/bin/env python

# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
# =============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import signal


def sigintHandler(signum, frame):
    print("exiting process")
    exit(-1)


def _query_visible_devices():
    r"""Query visible devices."""
    visible_devices_str = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices_str:
        visible_devices_str = os.getenv("NVIDIA_VISIBLE_DEVICES", "")
    if not visible_devices_str or visible_devices_str == "void":
        return []
    if visible_devices_str != "all":
        try:
            return visible_devices_str.split(",")
        except:  # pylint: disable=bare-except
            logging.exception("Parse NVIDIA_VISIBLE_DEVICES failed:")
            return []
    query_devices_command = (
        "nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null"
    )
    try:
        with subprocess.Popen(
            query_devices_command,
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        ) as proc:
            return [d for d in iter(proc.stdout.readline, b"") if d]
    except (OSError, ValueError):
        return []


def launch(command):
    r"""Run command in subprocess."""
    visible_devices = _query_visible_devices()
    local_world_size_str = str(len(visible_devices))
    strategy = os.getenv("COLLECTIVE_STRATEGY", "hb")

    signal.signal(signal.SIGINT, sigintHandler)
    signal.signal(signal.SIGHUP, sigintHandler)
    signal.signal(signal.SIGTERM, sigintHandler)

    if strategy == "hb":
        port = int(os.getenv("HB_RUN_BASE_PORT", "20001"))
        device_to_ports = []
        for d in visible_devices:
            device_to_ports.append([d, port])
            port += 1

        if len(device_to_ports) < 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["HB_OP_OPTIMIZATION_DISABLED"] = "1"
            if callable(command):
                command()
                return
            subprocess.check_call(command)
            return

        if len(device_to_ports) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            if callable(command):
                command()
                return
            subprocess.check_call(command)
            return

        tf_config = json.loads(os.getenv("TF_CONFIG", "{}"))
        if tf_config:
            task = tf_config["task"]
            task_type = task["type"]
            task_id = int(task["index"])
            cluster = tf_config["cluster"]
        else:
            task_type = "chief"
            task_id = 0
            cluster = {"chief": ["127.0.0.1:20000"]}

        workers = []
        if "chief" in cluster:
            workers.extend(cluster["chief"])
        if "worker" in cluster:
            workers.extend(cluster["worker"])
        worker_hosts = [w.split(":")[0] for w in workers]
        new_workers = [
            f"{h}:{p}" for h in worker_hosts for _, p in device_to_ports]
        new_cluster = cluster.copy()
        if "chief" in cluster:
            new_cluster["chief"] = [new_workers[0]]
            if len(new_workers) > 1:
                new_cluster["worker"] = new_workers[1:]
        else:
            new_cluster["worker"] = new_workers

        if task_type not in ("chief", "worker"):
            new_tf_config = {}
            new_tf_config["cluster"] = new_cluster
            new_tf_config["task"] = {}
            new_tf_config["task"]["type"] = task_type
            new_tf_config["task"]["index"] = task_id
            os.environ["TF_CONFIG"] = json.dumps(new_tf_config)
            os.environ["TF_TASK_TYPE"] = str(task_type)
            os.environ["TF_TASK_INDEX"] = str(task_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["HB_OP_OPTIMIZATION_DISABLED"] = "1"
            if callable(command):
                command()
            subprocess.check_call(command)
            return

        cpu_count = os.cpu_count()
        interop_threads = os.getenv("TF_NUM_INTEROP_THREADS", cpu_count)
        interop_threads_gpu = None
        if interop_threads:
            interop_threads_gpu = int(
                int(interop_threads) / len(device_to_ports))
            interop_threads_gpu = max(interop_threads_gpu, 4)
        intraop_threads = os.getenv("TF_NUM_INTRAOP_THREADS", cpu_count)
        intraop_threads_gpu = None
        if intraop_threads:
            intraop_threads_gpu = int(
                int(intraop_threads) / len(device_to_ports))
            intraop_threads_gpu = max(intraop_threads_gpu, 1)
        gpu_procs = {}
        gpu_envs = {}
        local_host = cluster[task_type][task_id].split(":")[0]
        for device, port in device_to_ports:
            gpu_addr = f"{local_host}:{port}"
            gpu_index = new_workers.index(gpu_addr)
            gpu_tf_config = {}
            gpu_tf_config["cluster"] = new_cluster
            gpu_tf_config["task"] = {}
            if "chief" in cluster:
                if gpu_index == 0:
                    gpu_tf_config["task"]["type"] = "chief"
                    gpu_tf_config["task"]["index"] = 0
                else:
                    gpu_tf_config["task"]["type"] = "worker"
                    gpu_tf_config["task"]["index"] = gpu_index - 1
            else:
                gpu_tf_config["task"]["type"] = "worker"
                gpu_tf_config["task"]["index"] = gpu_index
            gpu_env = os.environ.copy()
            gpu_env["TF_CONFIG"] = json.dumps(gpu_tf_config)
            gpu_env["TF_TASK_TYPE"] = gpu_tf_config["task"]["type"]
            gpu_env["TF_TASK_INDEX"] = str(gpu_tf_config["task"]["index"])
            gpu_env["CUDA_VISIBLE_DEVICES"] = device
            gpu_env["LOCAL_WORLD_SIZE"] = local_world_size_str
            if interop_threads_gpu:
                gpu_env["TF_NUM_INTEROP_THREADS"] = str(interop_threads_gpu)
            if intraop_threads_gpu:
                gpu_env["TF_NUM_INTRAOP_THREADS"] = str(intraop_threads_gpu)
            gpu_envs[device] = gpu_env

        if callable(command):
            procs = {}
            for device, _ in device_to_ports:

                def _target(env):
                    os.environ = env
                    command()

                proc = mp.Process(target=_target, args=(gpu_envs[device],))
                proc.start()
                procs[device] = proc
            done_procs = []
            for device, proc in procs.items():
                proc.join()
                done_procs.append(device)
                if proc.exitcode is not None and proc.exitcode != 0:
                    for term_gid, term_proc in procs.items():
                        if term_gid not in done_procs:
                            term_proc.terminate()
                            done_procs.append(term_gid)
                    if proc.exitcode < 0:
                        sys.exit(
                            f"Process {proc.pid} killed by "
                            f"{signal.Signals(-proc.exitcode).name}"
                        )
                    else:
                        sys.exit(
                            f"Process {proc.pid} exits unexpectedly: {proc.exitcode}"
                        )
            return

        for device, _ in device_to_ports:
            gpu_proc = subprocess.Popen(  # pylint: disable=consider-using-with
                command, env=gpu_envs[device], stdout=sys.stdout, stderr=sys.stderr
            )
            gpu_procs[gpu_proc.pid] = gpu_proc
        while True:
            if len(gpu_procs) < 1:
                break
            done_pids = []
            for pid, proc in gpu_procs.items():
                proc.poll()
                if proc.returncode is not None:
                    if proc.returncode == 0:
                        done_pids.append(pid)
                    else:
                        sys.exit(proc.returncode)
            for pid in done_pids:
                del gpu_procs[pid]
            time.sleep(1)

    elif strategy == "sok":

        def func_horovod_command(rank_size_str):
            horovod_command = ["horovodrun", "-np", rank_size_str]
            subprocess_command = []
            # shape like -H python main.py
            if isinstance(command, list):
                for cmd in command:
                    subprocess_command.extend(cmd.split(" "))
            elif isinstance(command, str):
                subprocess_command.append(command)
            horovod_command.extend(subprocess_command)
            return horovod_command

        port = int(os.getenv("HB_RUN_BASE_PORT", "20001"))
        device_to_ports = []
        for d in visible_devices:
            device_to_ports.append([d, port])
            port += 1

        if len(device_to_ports) < 1:
            logging.error("SOK mode currently not support CPU mode ")
            return

        if len(device_to_ports) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            horovod_command = func_horovod_command(local_world_size_str)
            subprocess.Popen(
                horovod_command, stdout=sys.stdout, stderr=sys.stderr)
            try:
                signal.pause()
            finally:
                proc.terminate()

                if proc.poll() is None:
                    proc.kill()

        cpu_count = os.cpu_count()
        interop_threads = os.getenv("TF_NUM_INTEROP_THREADS", cpu_count)
        interop_threads_gpu = None
        if interop_threads:
            interop_threads_gpu = int(
                int(interop_threads) / len(device_to_ports))
            interop_threads_gpu = max(interop_threads_gpu, 4)
        intraop_threads = os.getenv("TF_NUM_INTRAOP_THREADS", cpu_count)
        intraop_threads_gpu = None
        if intraop_threads:
            intraop_threads_gpu = int(
                int(intraop_threads) / len(device_to_ports))
            intraop_threads_gpu = max(intraop_threads_gpu, 1)

        envs = os.environ.copy()
        envs["TF_CONFIG"] = "{}"
        envs["TF_NUM_INTEROP_THREADS"] = str(interop_threads_gpu)
        envs["TF_NUM_INTRAOP_THREADS"] = str(intraop_threads_gpu)

        horovod_command = func_horovod_command(local_world_size_str)
        proc = subprocess.Popen(
            horovod_command, stdout=sys.stdout, stderr=sys.stderr)

        # 在父进程中等待子进程结束
        pid, status = os.wait()

        if os.WIFEXITED(status):
            print("subprocess exit normally with exit code: ",
                  os.WEXITSTATUS(status))
        elif os.WIFSIGNALED(status):
            print("subprocess exit abnormally with exit code:", os.WTERMSIG(status))

    else:
        logging.error("ENV `COLLECTIVE_STRATEGY` is unrecognized......")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", help="Command to launch script")
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments of the command"
    )
    args = parser.parse_args()
    launch([args.command] + args.args)

# pylint: disable=g-bad-file-header
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""async embedding stage class"""

from tensorflow.python.framework import ops
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.ops import prefetch

import os
import re

def async_embedding_mark_node(embedding_tensor):
    """ mark embedding lookup output
    Args: 
    embedding_tensor: output tensor of embedding lookup function, 
    usually it is consumed by hidden layers in the neural network.
    """
    ops.add_to_collections(ops.GraphKeys.ASYNC_EMBEDDING_OUTPUT_TENSORS, embedding_tensor)

class AsyncEmbeddingStage:
    """ async embedding stage is a helper class to add stage op after embedding
    look up operation, asynchronous embedding lookup and dense compute.
    """
    def __init__(self, threads_num = 1, capacity=1, checkpoint_dir = None):
        """ create async_embedding stage instance
        Args:
        threads_num: number of async_embedding prefetch threads.
        capacity: number of async_embedding prefetch buffer size.
        checkpoint_dir: path to dump graph.
        """
        self._threads_num = threads_num
        self._capacity = capacity
        self._checkpoint_dir = checkpoint_dir
        self._control_flow_ops = ['Switch', '_SwitchN', 'Merge', '_XlaMerge',
                                  'Enter', 'Exit']
        self._variable_ops = ['Variable', 'VariableV2', 'VarHandleOp',
                              'KvVarHandleOp', 'HashTableV2']
        self._variable_is_init_ops = ['IsVariableInitialized',
                                      'VarIsInitializedOp', 'KvVarIsInitializedOp']
        self._saver_ops = ['SaveV2']
        self._no_data_input_ops = self._variable_ops + ['Placeholder', 'PlaceholderV2', 'Const']
        self._boundary_ops = set()
        for tensor in ops.get_collection(ops.GraphKeys.ASYNC_EMBEDDING_OUTPUT_TENSORS):
            self._boundary_ops.add(tensor.op)
        
        self._dump_graph = \
            (os.getenv('ASYNC_EMBEDDING_DUMP_GRAPH', 'false').lower()== 'true')
        self._start_nodes = None
        self._node_control_outputs = {}
        self._active_nodes = set()
        self._stage_nodes = {}
        self._active_control_dependencies = {}
        self._stage_put_node = None
        self._stage_get_node = None

    def stage(self, graph):
        """ add async embedding stage node to graph
        """
        logging.info('async embedding stage begin')
        logging.info('async embedding thread num: ' + str(self._threads_num))
        logging.info('async embedding capacity: ' + str(self._capacity))
        
        self._save_graph(graph, "graph_before_async_embedding")
        self._find_start_node(graph)
        self._mark_nodes_status()
        self._pick_stage_nodes()
        self._perform_stage()
        self._check_graph()
        self._save_graph(graph, "graph_after_async_embedding")

        logging.info('async embedding stage end')

    def _save_graph(self, graph, name):
        if self._dump_graph and self._checkpoint_dir != None:
            logging.info("write {}.pbtxt".format(name))
            out_path = training_util.write_graph(graph.as_graph_def(add_shapes=True),
                                                 self._checkpoint_dir,
                                                 "{}.pbtxt".format(name))
            logging.info("write path is {}".format(out_path))

    def _is_control_flow_op(self, node):
        node_type = node.type
        for control_flow_node_type in self._control_flow_ops:
            if re.match(node_type, control_flow_node_type):
                return True
        return False

    def _is_variable_init_op(self, node):
        node_type = node.type
        for var_init_op in self._variable_is_init_ops:
            if re.match(node_type, var_init_op):
                return True
        return False

    def _is_saver_op(self, node):
        node_type = node.type
        for saver_op in self._saver_ops:
            if re.match(node_type, saver_op):
                return True
        return False

    def _is_no_data_input_op(self, node):
        input_size = len(node.inputs) + len(node.control_inputs)
        if input_size <= 0:
            return True
        return False

    def _find_start_node(self, graph):
        candidate_boundary_ops = self._boundary_ops.copy()
        if len(candidate_boundary_ops) <= 0:
            raise Exception("No boundary ops found")

        # travel all inputs node of boundary, remove the node if it is in
        # candidate_boundary_ops, and collect all no input node 
        is_visited = set()
        control_flow_nodes = set()
        no_data_input_nodes =set()
        is_find_io_staged_get = False
        for boundary_node in self._boundary_ops:
            visit_stack=[]
            visit_stack.append(boundary_node)
            while len(visit_stack) > 0:
                visit_node = visit_stack.pop()
                if visit_node in is_visited:
                    continue

                # check io staged get node is existed or not
                if (visit_node.type == "TensorBufferTake"):
                    is_find_io_staged_get = True
                
                # if meet node in candidate_boundary_ops and is not boundary_node
                # remove it from candidate_boundary_ops
                if visit_node != boundary_node and \
                   visit_node in candidate_boundary_ops:
                    candidate_boundary_ops.remove(visit_node)
                
                is_visited.add(visit_node)
                if self._is_control_flow_op(visit_node):
                    control_flow_nodes.add(visit_node)
                elif self._is_no_data_input_op(visit_node):
                    no_data_input_nodes.add(visit_node)
                for input_tensor in visit_node.inputs:
                    input_node = input_tensor.op
                    visit_stack.append(input_node)
                for control_input_node in visit_node.control_inputs:
                    visit_stack.append(control_input_node)

        if is_find_io_staged_get == False:
            raise Exception("Async Embedding: io staged is disabled, please check your code.")

        # 1. find all control_flow_op in graph, excluding ops in control_flow_nodes
        # 2. find all ops in graph who's type is in self._no_data_input_ops, excluding
        #    ops in no_data_input_nodes
        # 3. find all checking varialbe is initialized op in graph
        for node in graph.get_operations():
            if self._is_control_flow_op(node) and node not in control_flow_nodes:
                candidate_boundary_ops.add(node)
            elif self._is_no_data_input_op(node) and node not in no_data_input_nodes:
                candidate_boundary_ops.add(node)
            elif self._is_variable_init_op(node):
                candidate_boundary_ops.add(node)
        
        self._start_nodes = candidate_boundary_ops
        
    def _mark_nodes_status(self):
        active_stack = list(self._start_nodes)

        while len(active_stack) > 0:
            node = active_stack.pop()
            self._active_nodes.add(node)

            for output in node.outputs:
                for consumer in output.consumers():
                    if consumer not in self._active_nodes:
                        active_stack.append(consumer)

            # for control_output in self._get_node_control_outputs(node):
            for control_output in node._control_outputs:
                if control_output not in self._active_nodes:
                    active_stack.append(control_output)

        logging.debug('Async Embedding find active nodes = {}'.format(self._active_nodes))

    def _pick_stage_nodes(self):
        for active_node in self._active_nodes:
            # skip edge to saver op
            if self._is_saver_op(active_node):
                continue
            # find stage edge for data edge
            for active_node_input in active_node.inputs:
                if active_node_input.op not in self._active_nodes and \
                   active_node_input.op.type not in self._variable_ops:
                    # skip data edge from inactive variable node to active node
                    if active_node_input.op not in self._stage_nodes:
                        self._stage_nodes[active_node_input.op] = []
                    self._stage_nodes[active_node_input.op].append(active_node)

            # find stage edge for control dependency edge
            active_node_control_input_nodes = []
            for control_input_node in active_node.control_inputs:
                if control_input_node not in self._active_nodes:
                    active_node_control_input_nodes.append(control_input_node)
            if len(active_node_control_input_nodes) > 0:
                self._active_control_dependencies[active_node] = \
                    active_node_control_input_nodes

    def _get_input_ids(self, node, input):
        input_ids = []
        for input_id in range(len(node.inputs)):
            if node.inputs[input_id] == input:
                input_ids.append(input_id)
        return input_ids

    def _perform_stage(self):
        stage_outputs = {}
        stage_outputs_consumers = {}

        for stage_node in self._stage_nodes:
            stage_node_outputs = []
            stage_node_outputs_consumers = []
            for output in stage_node.outputs:
                output_consumers = set()
                for consumer in output.consumers():
                    if consumer in self._stage_nodes[stage_node]:
                        output_consumers.add(consumer)
                if len(output_consumers) <= 0:
                    continue
                stage_node_outputs.append(output)
                stage_node_outputs_consumers.append(output_consumers)

            if len(stage_node_outputs) <= 0:
                raise Exception('stage node must have output consumed by active nodes')

            stage_outputs[stage_node.name] = stage_node_outputs
            stage_outputs_consumers[stage_node.name] = stage_node_outputs_consumers
        with ops.colocate_with(list(self._stage_nodes.keys())[0]):
            stage_output_result = prefetch.staged(stage_outputs,
                                              num_threads=self._threads_num,
                                              capacity=self._capacity,
                                              timeout_millis=1000*60*60*3,
                                              closed_exception_types=\
                                              (errors.OutOfRangeError,))

        need_update_ops = []
        stage_op_parsed = False
        for stage_node_name in stage_outputs:
            stage_node_source_outputs = stage_outputs[stage_node_name]
            stage_node_target_outputs = stage_output_result[stage_node_name]
            stage_node_outputs_consumers = stage_outputs_consumers[stage_node_name]

            if stage_node_source_outputs is None or \
               stage_node_target_outputs is None or \
               stage_node_outputs_consumers is None or \
               len(stage_node_source_outputs) != len(stage_node_target_outputs) or \
               len(stage_node_source_outputs) != len(stage_node_outputs_consumers):
               raise Exception('mismatch stae input and output')

            for (source_output, target_output, output_consumers) in \
                zip(stage_node_source_outputs, stage_node_target_outputs,
                    stage_node_outputs_consumers):
                if not stage_op_parsed:
                    self._stage_put_node = None
                    for consumer in source_output.consumers():
                        if consumer.type == 'TensorBufferPut':
                            self._stage_put_node = consumer
                            logging.info('async embedding stage_put_node: {}'.format(consumer.name))
                            break
                    if self._stage_put_node is None:
                        raise Exception('no stage put node is found')
                    
                    self._stage_get_node = target_output.op.inputs[0].op
                    logging.info('async embedding stage_get_node: {}'.format(self._stage_get_node.name))

                    if self._stage_get_node.type != 'TensorBufferTake':
                        raise Exception('async embedding except unstage, '
                                        'actual: {}'.format(self._stage_get_node.type))
                    stage_op_parsed = True

                for consumer in output_consumers:
                    need_update_ops.append((consumer, source_output, target_output))

        # update data edges
        for need_update_op in need_update_ops:
            op = need_update_op[0]
            source_tensor = need_update_op[1]
            target_tensor = need_update_op[2]

            input_ids = self._get_input_ids(op, source_tensor)
            if len(input_ids) <= 0:
                raise Exception('no input id matches input tensor {}.format(source_tensor)')
            if len(input_ids) > 1:
                logging.warning('{} input ids match input tensor {}'.format(
                    len(input_ids), source_tensor))
            for input_id in input_ids:
                op._update_input(input_id, target_tensor)

        # switch active node control input from inactive node to stage get node,
        control_outputs = set()
        for active_node, active_control_input_nodes in \
            self._active_control_dependencies.items():
            control_inputs = []
            control_inputs.extend(active_node.control_inputs)
            for control_input_node in active_control_input_nodes:
                control_inputs.remove(control_input_node)
                control_outputs.add(control_input_node)
            control_inputs.append(self._stage_get_node)
            active_node._control_inputs = control_inputs
        
        # switch inactive node control output from active node to stage put node
        control_outputs = list(control_outputs)
        control_outputs.extend(self._stage_put_node.control_inputs)
        self._stage_put_node._control_inputs = control_outputs

    # travel from stage get node, we should not meet any inactive node
    def _find_node_cycle_dependent_on_stage(self, node, visited_set=set()):
        if node not in self._active_nodes and \
           node != self._stage_get_node and \
           node.type == "Identity" and node.inputs[0].op != self._stage_get_node:
            logging.error('{} is in inactive nodes.'.format(node.name))
            return [node]

        for data_input in node.inputs:
            input_node = data_input.op
            if input_node not in self._active_nodes and \
               input_node.type != self._stage_get_node and \
               input_node.type not in self._variable_ops and \
               input_node.type == "Identity" and input_node.inputs[0].op != self._stage_get_node:
                # skip data edge from inactive variable node to active node
                return [node]
        
        for control_input in node.control_inputs:
            if control_input != self._stage_get_node and \
               control_input not in self._active_nodes:
                logging.error("{} is control dep on inactive node {}".format(
                    node.name, control_input.name))
                return [node]

        for output in node.outputs:
            for consumer in output.consumers():
                if consumer in visited_set:
                    continue
                visited_set.add(consumer)
                sub_nodes = self._find_node_cycle_dependent_on_stage(consumer, visited_set)
                if sub_nodes:
                    nodes = [node]
                    nodes.extend(sub_nodes)
                    return nodes
        return None

    def _print_one_log(self, tag, msg):
      logging.info("async embedding {}: {}".format(tag, msg))

    def _print_ops_path(self, tag, ops):
      path = ""
      for op in ops:
        path += op.name
        path += "("
        path += op.node_def.op
        path += ")"
        path += " --> "
        self._print_one_log(tag, path)
    
    def _check_graph(self):
        visited_nodes = set()
        nodes = self._find_node_cycle_dependent_on_stage(self._stage_get_node, visited_nodes)
        if nodes:
            self._print_ops_path('find node cycle dependent on stage', nodes)
            raise Exception('check graph failed, find node cycle dependent on stage')
        

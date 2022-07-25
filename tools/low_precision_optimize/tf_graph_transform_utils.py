# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from simple_graph import SimpleGraph, SimpleNode
from tensorflow.python.framework import tensor_util


# [Basic graph utils]
def get_node_name_parts_from_input(input_name: str) -> Tuple[str, str, str]:
    input_parts = input_name.split(":")
    if len(input_parts) < 2:
        suffix = ""
    else:
        suffix = ":" + input_parts[1]
    node_name = input_parts[0]
    if node_name[0] == "^":
        prefix = "^"
        node_name = node_name[1:]
    else:
        prefix = ""
    return prefix, node_name, suffix


def get_node_name_from_input(input_name: Optional[str]) -> str:
    if input_name is None:
        return ""
    _, node_name, _ = get_node_name_parts_from_input(input_name)
    return node_name


def get_canonical_input_name(input_name: str) -> str:
    prefix, node_name, suffix = get_node_name_parts_from_input(input_name)
    suffix = ":0" if suffix == "" else suffix
    return prefix + node_name + suffix


def get_const_value(node: tf.NodeDef) -> np.ndarray:
    # Alternatively
    # tf.contrib.util.constant_value(tensor) will get a tensor"s constant value
    return tensor_util.MakeNdarray(node.attr["value"].tensor)


def get_const_value_by_name(
    graph_def: tf.GraphDef, name: str, simple_graph: Optional[SimpleGraph] = None
) -> np.ndarray:
    if simple_graph:
        node = get_node_by_name(graph_def, simple_graph, name)
        return get_const_value(node)
    else:
        node_name = get_node_name_from_input(name)
        founds = [nd for nd in graph_def.node if nd.name == node_name]
        if len(founds) == 0:
            error_msg = "Unknown node name: {}".format(node_name)
            raise Exception(error_msg)
        return get_const_value(founds[0])


# deprecated: use SimpleGraph.get_node_by_name instead
def get_node_by_name(
    graph_def: Optional[tf.GraphDef], simple_graph: SimpleGraph, name: str
) -> tf.NodeDef:
    return simple_graph.get_node_by_name(name)


# deprecated: use simple_graph.get_simple_node_by_name instead
def get_simple_node_by_name(simple_graph: SimpleGraph, name: str) -> SimpleNode:
    return simple_graph.get_simple_node_by_name(name)


# [Pattern matching]
def check_inputs(  # noqa: C901
    simple_graph: SimpleGraph,
    current_node: SimpleNode,
    pattern_nodes: Dict[str, SimpleNode],
    first_node: SimpleNode,
    matched_name_map: Dict[str, str],
) -> bool:
    # check op type
    if first_node.op != "*":
        matched_ops = [op.strip() for op in first_node.op.split("|")]
        if current_node.op not in matched_ops:
            return False
    # check node name
    if first_node.name in matched_name_map:
        if matched_name_map[first_node.name] != current_node.name:
            return False
    # check inputs
    if (len(first_node.inputs) == 1) and (first_node.inputs[0] == "*"):
        matched_name_map[first_node.name] = current_node.name
        return True
    # if inputs contains both unknown inputs and known inputs
    if (len(first_node.inputs) > 1) and ("*" in first_node.inputs):
        known_inputs = [name for name in first_node.inputs if name != "*"]
        start_idx = 0
        for key_name in known_inputs:
            matched = False
            if key_name.isdigit():
                matched = True
                continue
            for i in range(start_idx, len(current_node.inputs)):
                input_name = current_node.inputs[i]
                cur_input_node = simple_graph.get_simple_node_by_name(input_name)
                expected_input_op_str = (pattern_nodes[key_name].op).strip()
                if "|" in expected_input_op_str:
                    expected_input_ops = expected_input_op_str.split("|")
                else:
                    expected_input_ops = list([expected_input_op_str])
                if (cur_input_node.op in expected_input_ops) and (
                    check_inputs(
                        simple_graph,
                        cur_input_node,
                        pattern_nodes,
                        pattern_nodes[key_name],
                        matched_name_map,
                    )
                ):
                    matched = True
                    start_idx = i
            if not matched:
                return False
    # if all listed inputs are known inputs
    else:
        if len(current_node.inputs) != len(first_node.inputs):
            return False
        for i, input_name in enumerate(current_node.inputs):
            cur_input_node = simple_graph.get_simple_node_by_name(input_name)
            if first_node.inputs[i].isdigit():
                continue
            tmp_input_node = pattern_nodes[first_node.inputs[i]]
            if not check_inputs(
                simple_graph,
                cur_input_node,
                pattern_nodes,
                tmp_input_node,
                matched_name_map,
            ):
                return False
    matched_name_map[first_node.name] = current_node.name
    return True


def get_matched_pattern(
    simple_graph: SimpleGraph,
    pattern_nodes: Dict[str, SimpleNode],
    first_node_key: str,
    init_name_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    matched_name_maps = list()
    for i, node in enumerate(simple_graph.nodes):
        simple_node = get_simple_node_by_name(simple_graph, node.name)
        tmp_name_map = init_name_map.copy() if init_name_map else dict()
        if check_inputs(
            simple_graph,
            simple_node,
            pattern_nodes,
            pattern_nodes[first_node_key],
            tmp_name_map,
        ):
            matched_name_maps.append(tmp_name_map)
    return matched_name_maps


def remove_underscore_class_attr(graph_def: tf.GraphDef):
    for i, node in enumerate(graph_def.node):
        if '_class' in node.attr.keys():
            node.attr.pop('_class')

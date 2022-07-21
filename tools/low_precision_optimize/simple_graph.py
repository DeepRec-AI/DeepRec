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

import logging
from typing import Dict, List, Optional, Set

import tensorflow as tf


def get_canonical_tensor_name(name: str) -> str:
    """
    Legal tensor names are like: name, ^name, or name:digits. Please refert to:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/graph/tensor_id.cc#L35
    """
    parts = name.split(":")
    is_control_input = name.startswith("^")
    if len(parts) == 1:
        suffix = "" if is_control_input else ":0"
        return name + suffix
    elif len(parts) == 2 and parts[1].isdecimal() and not is_control_input:
        return name
    else:
        raise Exception(f"Invalid tensor name: {name}")


def tensor_name_to_node_name(tensor_name: str) -> str:
    return tensor_name.strip().strip("^").split(":")[0]


class SimpleNode:
    def __init__(
        self,
        name: str = "",
        op: str = "",
        inputs: List[str] = [],
        output_nodes: List[str] = [],
        tensors: Dict[str, List[str]] = {},
    ):
        self.name = name
        self.op = op
        self.inputs = inputs
        # Input tensors.
        self.inputs_tensors = [get_canonical_tensor_name(n) for n in inputs]
        # Output nodes.
        self.output_nodes = output_nodes.copy()
        # Mapping from output tensor name to list of nodes that consume this tensor.
        self.tensors = tensors.copy()

    @property
    def num_inputs(self) -> int:
        return len(self.inputs_tensors)

    @property
    def num_outputs(self) -> int:
        return len(self.output_nodes)

    @property
    def num_tensors(self) -> int:
        return len(self.tensors)

    @property
    def input_nodes(self) -> List[str]:
        return [tensor_name_to_node_name(inp) for inp in self.inputs_tensors]

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SimpleNode):
            return False
        return (
            self.name == o.name
            and self.op == o.op
            and self.inputs_tensors == o.inputs_tensors
            and self.output_nodes == o.output_nodes
            and self.tensors == o.tensors
        )

    def __str__(self) -> str:
        s = ""
        s += "name          : {}\n".format(self.name)
        s += "op            : {}\n".format(self.op)
        s += "inputs_tensors: {}\n".format(self.inputs_tensors)
        s += "ouput_nodes   : {}\n".format(self.output_nodes)
        s += "tensors       : {}\n".format(self.tensors)
        return s


class SimpleGraph:
    def __init__(self, graph_def: tf.GraphDef):
        self._nodes = [
            SimpleNode(name=n.name, op=n.op, inputs=list(n.input))
            for n in graph_def.node
        ]
        self._name2index = {n.name: i for i, n in enumerate(graph_def.node)}
        self._graph_def = graph_def

        for node in graph_def.node:
            for inp in node.input:
                inp_node_name = tensor_name_to_node_name(inp)
                inp_tensor_name = get_canonical_tensor_name(inp)
                if inp_node_name not in self._name2index:
                    raise Exception(f"SimpleNode {node.name}: Unknown input node {inp}")
                input_node = self._nodes[self._name2index[inp_node_name]]
                # update input node"s [output_node, ..] list
                input_node.output_nodes.append(node.name)
                # update input node"s {tensor: output_node, ..} dictionary
                #   TODO: we are missing Graph final output node"s tensors,
                #   but it is not possible to inspect how many tensors inside
                #   it, therefore we currently ignore it.
                if inp_tensor_name not in input_node.tensors:
                    input_node.tensors[inp_tensor_name] = [node.name]
                else:
                    input_node.tensors[inp_tensor_name].append(node.name)

    @property
    def num_nodes(self) -> int:
        """Get total number of nodes in graph."""
        return len(self._nodes)

    @property
    def nodes(self) -> List[SimpleNode]:
        """Get all nodes in graph."""
        return self._nodes

    def name2index(self, name: str) -> int:
        """Get index of node."""
        if name not in self._name2index:
            error_msg = "Node {} not exists".format(name)
            logging.error(error_msg)
            raise Exception(error_msg)
        return self._name2index[name]

    def node(self, idx: int) -> SimpleNode:
        """Get node with given index."""
        if idx >= len(self._nodes):
            error_msg = "Node index {} out of range".format(idx)
            logging.error(error_msg)
            raise Exception(error_msg)
        return self._nodes[idx]

    def name2node(self, name: str) -> SimpleNode:
        """Get node by name."""
        return self.node(self._name2index[name])

    def input_nodes(self, blacklist: List = ["Const"]) -> List[str]:
        """Get names of input nodes"""
        return [
            n.name for n in self._nodes if n.num_inputs == 0 and n.op not in blacklist
        ]

    def output_nodes(self) -> List[str]:
        """Get names of output nodes, which are those without downstream."""
        return [n.name for n in self._nodes if n.num_outputs == 0]

    def input_nodes_index(self, node_idx: int) -> List[int]:
        """Get indexes of input nodes for node of given index."""
        return [self._name2index[n] for n in self._nodes[node_idx].input_nodes]

    def get_simple_node_by_name(self, name: str) -> SimpleNode:
        node_name = tensor_name_to_node_name(name)
        if node_name not in self._name2index:
            raise Exception(f"Unknown node name: {node_name}")
        return self.node(self.name2index(node_name))

    def get_node_by_name(self, name: str) -> tf.NodeDef:
        node_name = tensor_name_to_node_name(name)
        if node_name not in self._name2index:
            raise Exception(f"Unknown node name: {node_name}")
        idx = self._name2index[node_name]
        if idx >= len(self._graph_def.node):
            raise Exception(f"Unknown node name: {node_name}")
        return self._graph_def.node[idx]

    def topological_sort(self, reverse: bool = False) -> List[int]:
        """Sort given SimpleGraph in topological order.

        Parameters
        ----------
        reverse : bool = False
            Set True to list op from output to input.

        Returns
        -------
        List[int]
            Index of graph node in topological order.
        """
        ready = []
        pending_count = []
        ordered = []
        # Parse the inputs for each node
        for i, node in enumerate(self._nodes):
            if node.op == "Merge" or node.op == "RefMerge":
                num_control_edges = sum(
                    1 for inp in node.inputs_tensors if inp.startswith("^")
                )
                pending_count.append(num_control_edges + 1)
            else:
                pending_count.append(len(node.inputs_tensors))
            if len(node.inputs_tensors) == 0:
                ready.append(i)
        processed = 0
        # Process the NodeDefs in topological order
        # Code above sets this up by filling in ready_ with nodes that have no
        # inputs, pending_counts_ with the number of inputs for each node and
        # outputs_ with the outputs of each node
        while len(ready) != 0:
            o = ready.pop(-1)
            ordered.append(o)
            processed += 1
            # Update pending_count for outputs.
            for out in self._nodes[o].output_nodes:
                pending_count[self._name2index[out]] -= 1
                if pending_count[self._name2index[out]] == 0:
                    ready.append(self._name2index[out])
        if processed < self.num_nodes:
            raise Exception(f"{self.num_nodes-processed} nodes in a cycle.")
        if reverse:
            ordered.reverse()
        return ordered

    def is_reachable(self, src_idx: int, target_idx: Set[int]) -> bool:
        """Check if any nodes with index in `target_idx` can be reached from `src_idx` node.

        Parameters
        ----------
        src_idx : int
            Index of source node.
        target_idx : Set[int]
            Indexes of target nodes.

        Returns
        -------
        bool
            True if any node in `target_idx` is reachable from source node.
        """
        if self.get_reachable(target_idx, src_idx):
            return True
        return False

    def get_reachable(
        self, target_idx: Set[int], start_at: Optional[int] = None
    ) -> Set[int]:
        """Get all nodes that can reach to the target_idx.

        Parameters
        ----------
        target_idx : Set[int]
            The index of target node.
        start_at : Optional[int] = None
            This option is to check if a start point can reach any nodes in the
            target_idx set.

        Returns
        -------
        Set[int]
            A set of the index of the nodes that can reach to any of the target nodes.
        """
        stack = list(target_idx)
        reachable = set()
        while len(stack) > 0:
            idx = stack.pop(-1)
            if idx in reachable:
                continue
            reachable.add(idx)
            if start_at is not None and idx == start_at:
                return reachable
            stack.extend(
                [self._name2index[inp] for inp in self._nodes[idx].input_nodes]
            )
        if start_at is not None:
            return set()
        return reachable

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_GRAPH_GRAPH_PARTITION_H_
#define TENSORFLOW_CORE_GRAPH_GRAPH_PARTITION_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

struct PartitionOptions {
  // A function that returns a location for the execution of a given
  // Node.
  typedef std::function<string(const Node*)> NodeToLocFunc;
  NodeToLocFunc node_to_loc = nullptr;

  // A function that returns a unique graph node name with the given
  // prefix.
  typedef std::function<string(const string&)> NewNameFunc;
  NewNameFunc new_name = nullptr;

  // A function that returns the incarnation of a device given the
  // device's fullname. If not found, GetIncarnationFunc should return
  // kIllegalIncarnation.
  static const uint64 kIllegalIncarnation = 0;
  typedef std::function<uint64(const string&)> GetIncarnationFunc;
  GetIncarnationFunc get_incarnation = nullptr;

  // If specified, flib_def defines a function library that should be
  // partitioned and replicated into each resulting partition graphs.
  const FunctionLibraryDefinition* flib_def = nullptr;

  // True if all the control flow "code" has already been added. The
  // control flow code needs to be added when we still have the entire
  // graph before any partitioning. So this flag should be false for
  // the first partitioning but true for all subsequent partitioning.
  //
  // TODO(yuanbyu): We could also make the addition of the control
  // flow code incremental based on 'node_to_loc'. This makes the
  // communication a broadcast tree, which could be more efficient when
  // the number of participating devices is large.
  bool control_flow_added = false;

  // A function that returns the data type into which the tensor
  // should be cast before sent over the wire.
  typedef std::function<DataType(const Edge*)> ShouldCastFunc;
  ShouldCastFunc should_cast = nullptr;

  // Schedule the execution of the recvs based on their start times
  // computed by some scheduling algorithm. The recvs are divided into
  // epochs based on their start times. A recv is enabled only when
  // execution reaches its epoch - N for some predefined N.
  bool scheduling_for_recvs = false;
  // The start time for each node in the graph computed by some scheduling
  // algorithm. If 'need_to_record_start_times' is true, we record them
  // in the graph as a node attribute.
  bool need_to_record_start_times = false;
  std::vector<Microseconds> start_times;

  // Fuse recv ops or not
  bool tensor_fuse = false;
};

// A map used to store memory types for the inputs/outputs of every node.
// The key is a pair of ints consisting of a node id and input/output index.
// TODO(power): migrate back to std::pair when absl::Hash is fixed for MSVC.
struct NodePort {
  int node_id;
  int index;

  friend bool operator==(const NodePort& x, const NodePort& y) {
    return x.node_id == y.node_id && x.index == y.index;
  }

  template <typename H>
  friend H AbslHashValue(H h, const NodePort& c) {
    return H::combine(std::move(h), c.node_id, c.index);
  }
};

typedef absl::flat_hash_map<NodePort, MemoryType> MemoryTypeMap;

// We collect the following information about the graph before performing
// graph partitioning.
struct GraphInfo {
  std::vector<DeviceType> device_types;
  MemoryTypeMap input_types;
  MemoryTypeMap output_types;
  std::vector<ControlFlowInfo> cf_info;
};

// Partition "input" graph into a set of graphs, one per location.
// The location for node n is derived by calling opts.node_to_loc(n).
// New nodes added by Partition use "opts.new_name(old_name)" to
// generate node names.
//
// Stores the partitions in *partitions.
Status Partition(const PartitionOptions& opts, Graph* input,
                 std::unordered_map<string, GraphDef>* partitions);

Status PartitionWithTensorFuse(const PartitionOptions& opts, Graph* g,
                               std::unordered_map<string, GraphDef>* partitions);

// Add control edges to the partitions to control the ordering
// and timing of the recv nodes based on the start times calculated
// using some scheduling algorithm.
Status AddControlEdges(const PartitionOptions& opts,
                       std::unordered_map<string, GraphDef>* partitions);

// Add an input to dst that comes from the "src_slot" output of the
// node named by "src_name".
void AddInput(NodeDef* dst, StringPiece src_name, int src_slot);

// Each participating device needs to decide a) if there is a next iteration,
// and b) if the loop terminates. We take the approach to encode this control
// flow logic in the dataflow graph. There are at least two possible encodings.
// In a completely decentralized encoding, the participants communicate peer
// to peer. The other encoding uses a frame leader (the participant who owns
// the pivot termination predicate) to broadcast the termination condition to
// all the participants. For now we take the latter because it is simpler.
//
// TODO(yuanbyu): The correctness of this construction is rather subtle. I got
// it wrong many times so it would be nice to write a proof to be sure.
Status AddControlFlow(const PartitionOptions& opts, Graph* g, GraphInfo* g_info);

// Build memory and device type info for every node in the graph.
// TODO(yuanbyu): It might be simpler if we convert MemoryType to
// DeviceType for the inputs/outputs of each node.
Status BuildMemoryDeviceInfo(const Graph& g, GraphInfo* info);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_PARTITION_H_

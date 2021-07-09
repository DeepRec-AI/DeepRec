#ifndef TENSORFLOW_CORE_GRAPH_DIST_GRAPH_PARTITION_H_
#define TENSORFLOW_CORE_GRAPH_DIST_GRAPH_PARTITION_H_

#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_partition.h"

namespace tensorflow {

// Define policy that use Send/Recv instead of RunGraph.
// Eg: RunGraph can't solve cycle problem, here we use
//     _Send/_Recv to solve it.
// User can define other policy.
class SendRecvPolicy {
 public:
  SendRecvPolicy(Graph* graph);
  virtual void AddUserDefinedPolicy();
  virtual bool UseSendRecvMode(Node* src, Node* dst);
  virtual ~SendRecvPolicy();

 private:
  void UseDefaultPolicy();

 protected:
  Graph* graph_;
  // Default policy, indicate ps1 will use Send/Recv with other ps that 
  // defined in std::unordered_set<std::string>, the value of the unordered_set
  // is the location of the other ps. User can define other policy.
  std::unordered_map<std::string, std::unordered_set<std::string>> _use_send_recv_loc;
};

class DetectPsToPsCyclePolicy : public SendRecvPolicy {
 public:
  DetectPsToPsCyclePolicy(Graph* graph);
  virtual ~DetectPsToPsCyclePolicy();
  virtual void AddUserDefinedPolicy();

 private:
};

class SubGraph {
public:
  typedef std::vector<std::pair<std::string, const Edge*>> KeyEdgeVec;
  SubGraph(const std::string &loc) :
    loc_(loc), only_variable_(false), use_send_recv_(false) {}
  SubGraph() : only_variable_(false), use_send_recv_(false) {}
  const std::string& GetLoc() const { return loc_; }
  void SetLoc(const std::string &loc) { loc_ = loc; }

  void AddNode(const Node* node) {
    if (node_id_set_.find(node->id()) != node_id_set_.end()) {
      return;
    }
    nodes_.push_back(node);
    node_id_set_.insert(node->id());
  }

  void AddInputEdge(const std::string& feed_name, const Edge* edge) {
    input_edges_.push_back({feed_name, edge});
  }

  void AddInputs(const std::vector<std::pair<std::string, const Edge*> > &input_edges) {
    input_edges_.insert(input_edges_.end(), input_edges.begin(), input_edges.end());
  }

  void AddOutputEdge(const std::string& fetch_name, const Edge* edge) {
    output_edges_.push_back({fetch_name, edge});
  }
  void AddOutputs(const std::vector<std::pair<std::string, const Edge*> > &output_edges) {
    output_edges_.insert(output_edges_.end(), output_edges.begin(), output_edges.end());
  }

  bool NodeExisted(int id) {
    return node_id_set_.find(id) != node_id_set_.end();
  }
  void SetOnlyVariable() { only_variable_ = true;}
  bool IsOnlyVariable() const { return only_variable_;}
  const KeyEdgeVec& GetInputEdges() const { return input_edges_; }
  const KeyEdgeVec& GetOutputEdges() const { return output_edges_; }

  const std::vector<const Node*>& GetNodes() const { return nodes_; }

  GraphDef& GetGraphDef() { return graph_def_; }

  const GraphDef& GetGraphDef() const { return graph_def_; }

  void Extend(const SubGraph &sub_graph) {
    for (const Node* node : sub_graph.GetNodes()) {
      if (loc_ == "") {
        loc_ = sub_graph.GetLoc();
      }
      AddNode(node);
    }
  }

  void SetGraphHandle(const std::string &handle) {
    graph_handle_ = handle;
  }

  const std::string& GetGraphHandle() const {
    return graph_handle_;
  }

  void CompleteVariables(const PartitionOptions& opts);

  std::string GetDeviceName() {
    if (nodes_.size() == 0) {
      LOG(WARNING) << "nodes size is 0, loc:" << loc_;
      return "";
    }
    return nodes_[0]->assigned_device_name();
  }

  void SetSendRecvFlag(bool has) {
    use_send_recv_ = has;
  }

  const bool GetSendRecvFlag() const {
    return use_send_recv_;
  }

private:
  std::string loc_;
  std::vector<const Node*> nodes_;
  GraphDef graph_def_;
  std::string graph_handle_;
  KeyEdgeVec input_edges_;
  KeyEdgeVec output_edges_;
  std::unordered_set<int> node_id_set_;
  bool only_variable_;
  // Use _Send/_Recv instead of RunGraph
  bool use_send_recv_;
};

typedef std::pair<std::string, int> InputSrcKey;

class GraphPartitionerBase {
 public:
  GraphPartitionerBase(
      const PartitionOptions& popts, Graph* g, bool zero_copy, bool use_fuse_recv,
      const std::function<bool (const std::string &)> &is_main_loc_func,
      SendRecvPolicy* user_define_policy);

  virtual ~GraphPartitionerBase() {
    if (use_default_policy_) {
      delete send_recv_policy_;
    }
  }

 public:
  Status CompleteSubGraphs(std::vector<SubGraph> *sub_graphs);
  Status CompleteSubGraphsV2(std::vector<SubGraph> *sub_graphs);

  Status CompleteMainGraph(const std::vector<SubGraph> &sub_graphs,
                           SubGraph *main_graph);
  Status CompleteMainGraphV2(
      const std::vector<SubGraph> &sub_graphs,
      SubGraph *worker_graph);

 protected:
  bool ShouldUseSendRecvMode(Node* src, Node* dst);

  Status SplitGraphInternal(std::vector<SubGraph> *sub_graphs,
                            SubGraph *main_graph,
                            bool needResetSwitchOp);
  Status SplitGraphInternalV2(std::vector<SubGraph> *sub_graphs,
                              SubGraph *main_graph,
                              bool needResetSwitchOp);

  std::unordered_set<const Node*> GetClusteringNodes(
      std::unordered_set<const Node*> *ready,
      std::unordered_set<const Node*> *nodes);

  Status CompleteSubGraph(SubGraph &sub_graph, int graph_idx);
  Status CompleteSubGraphV2(SubGraph &sub_graph, int graph_idx,
                            std::unordered_map<const Node*, int> &node_to_subgraph_id);

  Status ProcessSubNodeInputsV2(
            const Node* node,
            const std::string &loc,
            NodeDef *node_def,
            GraphDef *graph_def,
            std::map<InputSrcKey, NodeDef*> *input_src_nodes_map,
            std::map<InputSrcKey, NodeDef*> *local_input_map,
            std::map<InputSrcKey, NodeDef*> *local_ref_input_map,
            std::vector<std::pair<std::string, const Edge*>> *boundary_input_edges,
            SubGraph &sub_graph,
            std::unordered_map<const Node*, int> &node_to_subgraph_id);

  Status ProcessSubNodeInputs(
            const Node* node,
            const std::string &loc,
            NodeDef *node_def,
            GraphDef *graph_def,
            std::map<InputSrcKey, NodeDef*> *input_src_nodes_map,
            std::vector<std::pair<std::string, const Edge*>> *boundary_input_edges,
            bool& has_direct_edge);

  bool UseFuseRecv(SubGraph &sub_graph, int graph_idx,
                   NodeDef** new_fuse_recv_node,
                   std::unordered_map<std::string, int>& src_to_slot);

  bool UseFuseRecvV2(SubGraph &sub_graph, int graph_idx,
                     NodeDef** new_fuse_recv_node,
                     std::unordered_map<std::string, int>& src_to_slot);

  bool UseFuseRecvInternal(SubGraph &sub_graph, int graph_idx,
                           NodeDef** new_fuse_recv_node,
                           std::unordered_map<std::string, int>& src_to_slot,
                           bool is_version1);

  // Enable fuse recv op
  Status ProcessSubNodeInputs(
            const Node* node,
            const std::string &loc,
            NodeDef *node_def,
            GraphDef *graph_def,
            std::map<InputSrcKey, NodeDef*> *input_src_nodes_map,
            std::vector<std::pair<std::string, const Edge*>> *boundary_input_edges,
            bool& has_direct_edge,
            NodeDef* fuse_recv_node,
            std::unordered_map<std::string, int>& key_to_idx);

  // Enable fuse recv op
  Status ProcessSubNodeInputsV2(
            const Node* node,
            const std::string &loc,
            NodeDef *node_def,
            GraphDef *graph_def,
            std::map<InputSrcKey, NodeDef*> *input_src_nodes_map,
            std::map<InputSrcKey, NodeDef*> *local_input_map,
            std::map<InputSrcKey, NodeDef*> *local_ref_input_map,
            std::vector<std::pair<std::string, const Edge*>> *boundary_input_edges,
            NodeDef* fuse_recv_node,
            std::unordered_map<std::string, int>& key_to_idx,
            SubGraph &sub_graph,
            std::unordered_map<const Node*, int> &node_to_subgraph_id);

  NodeDef* InsertLocalRecvNode(
      const Edge* in_edge,
      std::map<InputSrcKey, NodeDef*> *input_src_nodes_map,
      std::map<InputSrcKey, NodeDef*> *ref_input_src_nodes_map,
      GraphDef *graph_def,
      std::unordered_map<const Node*, int> &node_to_subgraph_id);
 
  Status CreateLocalRecvNode(
      const Edge *in_edge,
      NodeDef *src_node_def,
      std::unordered_map<const Node*, int> &node_to_subgraph_id);

  NodeDef* InsertLocalSendNode(
      const Edge* out_edge,
      std::map<int, std::unordered_map<std::string, NodeDef*>>
          &local_send_nodes_map,
      std::map<int, std::unordered_map<std::string, NodeDef*>> 
          &local_ref_send_nodes_map,
      GraphDef *graph_def,
      std::unordered_map<const Node*, int> &node_to_subgraph_id);

  Status CreateLocalSendNode(
      const Edge* out_edge,
      NodeDef *send_node_def,
      GraphDef *graph_def,
      std::unordered_map<const Node*, int> &node_to_subgraph_id);
 
  Status ProcessSubNodeOutputsV2(
            const Node* node,
            const std::string &loc,
            NodeDef *node_def,
            GraphDef *graph_def,
            std::vector<std::pair<std::string, const Edge*>> *boundary_output_edges,
            SubGraph &sub_graph,
            std::unordered_map<const Node*, int> &node_to_subgraph_id);

  Status ProcessSubNodeOutputs(
            const Node* node,
            const std::string &loc,
            NodeDef *node_def,
            GraphDef *graph_def,
            std::vector<std::pair<std::string, const Edge*>> *boundary_output_edges,
            bool& has_direct_edge);

  Status ConstructSendNodeDef(
            const std::string &node_name,
            const std::string &send_device_name,
            const std::string &recv_device_name,
            const std::string &input_node_name,
            const int &input_idx,
            const std::string &tensor_name,
            const DataType &tensor_type,
            bool client_terminated,
            NodeDef *node_def);

  Status ConstructRecvNodeDef(const PartitionOptions &opts,
                              const std::string &node_name,
                              const std::string &send_device_name,
                              const std::string &recv_device_name,
                              const std::string &tensor_name,
                              const DataType &tensor_type,
                              bool client_terminated,
                              NodeDef *node_def);

  Status ProcessRunGraphInputs(const SubGraph &ps_graph,
                               const std::string &worker_device,
                               GraphDef *worker_graph_def,
                               NodeDef *run_graph_node_def,
                               std::map<InputSrcKey, NodeDef*> *bridge_nodes_map);

  void TryAddEdgeNode(
            const std::string &cur_loc, const Node *node,
            std::deque<const Node*> &nodes,
            std::vector<bool> &node_visited, SubGraph &cur_sub_graph);

  Status ProcessRunGraphOutputs(const SubGraph &ps_graph,
                                const std::string &worker_device,
                                GraphDef *worker_graph_def,
                                NodeDef *run_graph_node_def,
                                std::map<InputSrcKey, NodeDef*> *bridge_nodes_map,
                                std::map<int, NodeDef*> *added_nodes);

  bool IsSplitedIgnored(const std::string &cur_loc, const Node* node);

  Status ConstructNodeDef(const Node* node, NodeDef *node_def);

  Status ResetSwitchOpDevice();

  std::string GetWorkerDevice();

  NodeDef* DealWithSubNodeInputEdge(
            const Edge* in_edge,
            std::map<InputSrcKey, NodeDef*> *input_src_nodes_map,
            std::string *feed_key,
            GraphDef *graph_def);

  void MergeReadyPsSubGraphs(
            std::unordered_set<const SubGraph*> *ps_sub_graph_set,
            std::unordered_set<const Node*> *ready_nodes,
            std::vector<SubGraph> *merged_ps_sub_graphs);

  Status MergePsGraphs(
            const SubGraph &worker_sub_graph,
            const std::vector<SubGraph> &ps_sub_graphs,
            std::vector<SubGraph> *merged_ps_graphs);

  void RemoveReadyNodes(
      std::unordered_set<const Node*> *ready_nodes,
      std::unordered_set<const Node*> *not_ready_worker_nodes);

  virtual NodeDef* DealWithSubNodeOutputEdge(
            const Edge* out_edge,
            std::map<InputSrcKey, NodeDef*> &send_nodes_map,
            GraphDef *graph_def) = 0;

  virtual NodeDef* DealWithSubNodeOutputEdge(
            const Edge* out_edge,
            std::unordered_map<std::string, NodeDef*>& send_nodes_map,
            std::unordered_map<std::string, int>& send_nodes_index,
            GraphDef *graph_def, bool is_version1) = 0;

  virtual Status MakeInputSrcNode(const Edge *in_edge,
                                  NodeDef *src_node_def,
                                  std::string *feed_key) = 0;

protected:
  PartitionOptions opts_;
  Graph* graph_;
  std::function<bool (const std::string &)> is_main_loc_func_;
  SendRecvPolicy* send_recv_policy_;
  bool use_default_policy_;
  bool zero_copy_;
  bool use_fuse_recv_;
};

class TrainGraphPartitioner : public GraphPartitionerBase {
 public:
   TrainGraphPartitioner(const PartitionOptions& popts, Graph* g,
                         bool zero_copy, bool use_fuse_recv,
                         SendRecvPolicy* user_define_policy = nullptr)
     : GraphPartitionerBase(popts, g, zero_copy, use_fuse_recv,
                            [](const std::string loc){
                              return loc.find("worker") != std::string::npos;
                            }, user_define_policy) {}
  ~TrainGraphPartitioner() {}

  Status SplitGraph(SubGraph *worker_sub_graph,
                    std::vector<SubGraph> *ps_sub_graphs,
                    bool merge_ps_graph = true);

  Status SplitGraphV2(SubGraph *worker_sub_graph,
                      std::vector<SubGraph> *ps_sub_graphs);

 protected:
  NodeDef* DealWithSubNodeOutputEdge(
            const Edge* out_edge,
            std::map<InputSrcKey, NodeDef*> &send_nodes_map,
            GraphDef *graph_def) override;
  NodeDef* DealWithSubNodeOutputEdge(
            const Edge* out_edge,
            std::unordered_map<std::string, NodeDef*>& send_nodes_map,
            std::unordered_map<std::string, int>& send_nodes_index,
            GraphDef *graph_def, bool is_version1) override;

  Status MakeInputSrcNode(const Edge *in_edge,
                          NodeDef *src_node_def,
                          std::string *feed_key) override;

};

class InferGraphPartitioner : public GraphPartitionerBase {
 public:
  InferGraphPartitioner(const PartitionOptions& popts,
                        Graph* g,
                        const std::string &main_device_name,
                        SendRecvPolicy* user_define_policy = nullptr)
    : GraphPartitionerBase(
          popts, g, false, false,
          [main_device_name](const std::string &loc){
            return loc.find(main_device_name) != std::string::npos;
          }, user_define_policy) {}
  virtual ~InferGraphPartitioner(){}

  Status SplitGraph(SubGraph *main_sub_graph,
                    std::vector<SubGraph> *sub_graphs,
                    bool merge_ps_graph = true);

 protected:
  NodeDef* DealWithSubNodeOutputEdge(
            const Edge* out_edge,
            std::map<InputSrcKey, NodeDef*> &send_nodes_map,
            GraphDef *graph_def) override;
  NodeDef* DealWithSubNodeOutputEdge(
            const Edge* out_edge,
            std::unordered_map<std::string, NodeDef*>& send_nodes_map,
            std::unordered_map<std::string, int>& send_nodes_index,
            GraphDef *graph_def, bool is_version1) override;

  Status MakeInputSrcNode(const Edge *in_edge,
                          NodeDef *src_node_def,
                          std::string *feed_key) override;

};

std::string node_to_loc(const Node *node);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_DIST_GRAPH_PARTITION_H_


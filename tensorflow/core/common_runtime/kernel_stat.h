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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_KERNEL_STAT_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_KERNEL_STAT_H_

#include <atomic>
#include <memory>
#include <queue>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/graph_view.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace ExecutorInternal {

namespace {
static const std::string start_node_stats_step_env_name =
    "START_NODE_STATS_STEP";
static const std::string stop_node_stats_step_env_name =
    "STOP_NODE_STATS_STEP";
}

// Hold stats info
struct KernelStatsInfo {
  int64 op_start_time_ = 0;
  int64 op_stop_time_ = 0;
  // Add other info below
};

// Stores execution time information about the kernels in an executor's graph.
class KernelStats {
 public:
  KernelStats() : wait_to_collect_(true), collect_op_cost_(false),
      stop_counter_(0), counter_(0) {
    Status s = ReadInt64FromEnvVar(
        start_node_stats_step_env_name, 100, &start_step_);
    if (!s.ok()) {
      LOG(WARNING) << "Read START_NODE_STATS_STEP envrionment error. "
                 << s.error_message();
    }    
    s = ReadInt64FromEnvVar(
        stop_node_stats_step_env_name, 200, &stop_step_);
    if (!s.ok()) {
      LOG(WARNING) << "Read STOP_NODE_STATS_STEP envrionment error. "
                 << s.error_message();
    }    
    if (start_step_ > 0 && stop_step_ > 0 && 
        stop_step_ > start_step_) {
      collect_kernel_stats = true;
      VLOG(1) << "User collect node stats, start_step is " << start_step_
              << ", stop_step is " << stop_step_;
    }    
  }    

  void Initialize(const GraphView& gview,
                  const Graph* g) { 
    gv_ = const_cast<GraphView*>(&gview);
    g_ = const_cast<Graph*>(g);

    nodes_count_ = gview.num_nodes();
    is_expensive_.resize(gview.num_nodes());
    cost_estimates_ =
        absl::make_unique<std::atomic_uint_fast64_t[]>(gview.num_nodes());
    immutable_avg_cost_ =
        absl::make_unique<std::atomic<int64_t>[]>(gview.num_nodes());
    node_stats_count_ =
        absl::make_unique<std::atomic<int32_t>[]>(gview.num_nodes());
    task_count_ =
        absl::make_unique<std::atomic<int32_t>[]>(gview.num_nodes());
    for (int32_t i = 0; i < gview.num_nodes(); ++i) {
      if (gview.node(i)) {
        is_expensive_[i] =
            gview.node(i)->kernel && gview.node(i)->kernel->IsExpensive();
        cost_estimates_[i] = kInitialCostEstimateCycles;
        immutable_avg_cost_[i] = 0;
        node_stats_count_[i] = 0;
        task_count_[i] = 0;
      }
    }
  }

  // Returns true iff the given node is considered "expensive". The
  // executor uses this flag to optimize graph execution, for example
  // by "inlining" inexpensive kernels.
  bool IsExpensive(const NodeItem& node) const {
    return is_expensive_[node.node_id] &&
           (cost_estimates_[node.node_id].load(std::memory_order_relaxed) >
            kOpIsExpensiveThresholdCycles);
  }

  // Returns the value of kernel->IsExpensive().
  bool HasExpensiveMarker(const NodeItem& node) const {
    return is_expensive_[node.node_id];
  }

  // Updates the dynamic cost estimate, which is used to determine whether the
  // given node is expensive. The new cost estimate is a weighted average of
  // the old cost estimate and the latest cost. We only update cost estimates
  // for kernels for which IsExpensive() return true.
  void UpdateCostEstimate(const NodeItem& node, uint64 elapsed_cycles) {
    // N.B. Updates to `cost_estimate` are atomic but unlocked.  Simultaneous
    // updates may result in one or more updates being ignored.  This does not
    // affect correctness but may slow down the update frequency.
    std::atomic_uint_fast64_t& cost_estimate = cost_estimates_[node.node_id];
    auto prev_estimate = cost_estimate.load(std::memory_order_relaxed);

    uint64 new_estimate =
        ((kCostDecay - 1) * prev_estimate + elapsed_cycles) / kCostDecay;

    cost_estimate.store(new_estimate, std::memory_order_relaxed);
  }

  void CalculateAccumulativeCost() {
    std::queue<Node*> q;
    std::unordered_map<Node*, int> pending_childs;
    for (auto n : g_->nodes()) {
      pending_childs[n] = n->out_edges().size();
      if (n->out_edges().empty()) {
        q.push(n);
      }
    }

    immutable_accumulative_cost_.resize(nodes_count_);
    while (!q.empty()) {
      Node* curr = q.front();
      q.pop();
      immutable_accumulative_cost_[curr->id()] =
          immutable_avg_cost_[curr->id()];
      for (auto edge : curr->out_edges()) {
        int dest_id = edge->dst()->id();
        int64 tmp = immutable_avg_cost_[curr->id()] +
            immutable_accumulative_cost_[dest_id];
        if (immutable_accumulative_cost_[curr->id()] < tmp) {
          immutable_accumulative_cost_[curr->id()] = tmp;
        }
      }

      for (auto edge : curr->in_edges()) {
        if (--pending_childs[edge->src()] == 0) {
          q.push(edge->src());
        }
      }
    }
  }

  void StopCollection() {
    collect_op_cost_ = false;

    // 1.calculate average cost
    for (size_t i = 0; i < nodes_count_; ++i) {
      int32_t count = node_stats_count_[i];
      if (count > 0) {
        immutable_avg_cost_[i] = immutable_avg_cost_[i] / count;
      }
    }

    // 2.calculate accumulative op cost
    CalculateAccumulativeCost();

    // 3. calculate other metrics here

    collect_stats_done_ = true;
  }

  // Trace node info, for example execute time etc.
  void MaybeCollectKernelStats() {
    if (!collect_kernel_stats ||
        collect_stats_done_) return;

    if (collect_op_cost_) {
      auto current = counter_.fetch_add(1);
      if (current >= stop_step_) {
        int stop_counter = stop_counter_.fetch_add(1);
        if (stop_counter == 0) {
          StopCollection();
        }
      }
      return;
    }

    if (!wait_to_collect_) return;
    auto current = counter_.fetch_add(1);
    if (current == start_step_) {
      wait_to_collect_ = false;
      collect_op_cost_ = true;
    }
  }

  void StartCollectOp(const NodeItem* item, KernelStatsInfo* stat) {
    if (!collect_kernel_stats ||
        collect_stats_done_ ||
        !collect_op_cost_) {
      return;
    }

    stat->op_start_time_ = Env::Default()->NowNanos();
    task_count_[item->node_id] = 0;
  }

  void StopCollectOp(const NodeItem* item, KernelStatsInfo* stat) {
    if (!collect_kernel_stats ||
        collect_stats_done_ ||
        !collect_op_cost_) {
      return;
    }

    stat->op_stop_time_ = Env::Default()->NowNanos();
    if (item->node_id >= nodes_count_) {
      LOG(WARNING) << "Item node is exceed nodes_count_, "
                   << item->node_id << " VS " << nodes_count_;
    }

    immutable_avg_cost_[item->node_id] +=
        (stat->op_stop_time_ - stat->op_start_time_);

    node_stats_count_[item->node_id]++;
    // Collect Other info here

  }

  void OpScheduleTask(const NodeItem* item) {
    if (!collect_kernel_stats ||
        collect_stats_done_ ||
        !collect_op_cost_) {
      return;
    }
    task_count_[item->node_id]++;
  }

  int64 GetNodeCost(const NodeItem* item) {
    if (item->node_id >= nodes_count_) {
      LOG(WARNING) << "Item node is exceed nodes_count_, "
                   << item->node_id << " VS " << nodes_count_;
    }
    return immutable_avg_cost_[item->node_id];
  }

  int64 GetIntraCost(const NodeItem* item) {
    if (item->node_id >= nodes_count_) {
      LOG(WARNING) << "Item node is exceed nodes_count_, "
                   << item->node_id << " VS " << nodes_count_;
    }
    if (task_count_[item->node_id] == 0) {
      return immutable_avg_cost_[item->node_id];
    }
    return immutable_avg_cost_[item->node_id] / task_count_[item->node_id];
  }

  int64 GetOpAccumulativeCost(const NodeItem* item) {
    if (item->node_id >= nodes_count_) {
      LOG(WARNING) << "Item node is exceed nodes_count_, "
                 << item->node_id << " VS " << nodes_count_;
    }
    return immutable_accumulative_cost_[item->node_id];
  }

  const std::vector<int64>* GetAccumulativeCostArray() {
    return &immutable_accumulative_cost_;
  }

  const int64 GetNodeCount() {
    return nodes_count_;
  }

  bool CollectStatsDone() const {
    return collect_stats_done_;
  }

 private:
  // Initial time (in CPU cycles) we expect an operation to take.  Used to
  // determine whether an operation should be place in a threadpool.
  // Operations start out "expensive".
  static constexpr uint64 kInitialCostEstimateCycles = 100 * 1000 * 1000;
  static constexpr uint64 kOpIsExpensiveThresholdCycles = 8000;
  static constexpr uint64 kCostDecay = 10;

  std::vector<bool> is_expensive_;
  std::unique_ptr<std::atomic_uint_fast64_t[]> cost_estimates_;

  // step to start/stop trace node
  // User can set envrionment
  // 'START_NODE_STATS_STEP' and 'STOP_NODE_STATS_STEP'
  // to modify the value.
  int64 start_step_ = -1;
  int64 stop_step_ = -1;
  bool collect_kernel_stats = false;
  std::atomic<bool> wait_to_collect_;
  std::atomic<bool> collect_op_cost_;
  std::atomic<int> stop_counter_;
  bool collect_stats_done_ = false;
  std::atomic<int64_t> counter_;
  int64_t nodes_count_ = 0;
  // Average execution time of nodes
  std::unique_ptr<std::atomic<int64_t>[]> immutable_avg_cost_;
  std::unique_ptr<std::atomic<int32_t>[]> node_stats_count_;
  // The max total execute time of the graph execute path,
  // which from current node to the sink node.
  // Example:
  // A(1) -> B(1) -> C(1) --
  //   |                   | -> sink(0)
  //   --> D(3) ------------
  // the max total execute time of A is MAX(1+1+1+0, 1+3+0) = 4
  std::vector<int64> immutable_accumulative_cost_;

  // number of tasks scheduled by the operator to the thread pool
  std::unique_ptr<std::atomic<int32_t>[]> task_count_;

  GraphView* gv_ = nullptr; // not owned
  Graph* g_ = nullptr; // not owned
};

}  // end namespace ExecutorInternal
}  // end namespace tensorflow

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_KERNEL_STAT_H_

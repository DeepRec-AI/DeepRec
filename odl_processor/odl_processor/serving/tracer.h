#ifndef ODL_PROCESSOR_SERVING_TRACER_H
#define ODL_PROCESSOR_SERVING_TRACER_H

#include <fstream>
#include <iostream>
#include <string>
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace processor {

class Tracer {
 public:
  static Tracer* GetTracer() {
    static Tracer t;
    return &t;
  }

  Tracer() : tracing_(false), curr_step_(0) {}
  Tracer(int64_t start_step,
         int64_t interval_step, 
         int64_t tracing_count)
    : tracing_(true),
      next_tracing_step_(start_step),
      interval_step_(interval_step),
      tracing_count_(tracing_count),
      curr_step_(0) {
    limit_step_ = start_step +
        interval_step * tracing_count;
    PrintDebugString();
  }

  void SetParams(const int64_t start_step,
                 const int64_t interval_step,
                 const int64_t tracing_count) {
    tracing_ = true;
    next_tracing_step_ = start_step;
    interval_step_ = interval_step;
    tracing_count_ = tracing_count;
    limit_step_ = start_step +
        interval_step * tracing_count;
    PrintDebugString();
  }

  bool NeedTracing() {
    if (tracing_ && curr_step_ < limit_step_) {
      int64_t s = curr_step_.fetch_add(1, std::memory_order_relaxed);
      if (s == next_tracing_step_) {
        mutex_lock lock(mu_);
        next_tracing_step_ += interval_step_;
        return true;
      }
    }

    return false;
  }

  void GenTimeline(tensorflow::RunMetadata& run_metadata) {
    static std::atomic<int> counter(0);
    int index = counter.fetch_add(1, std::memory_order_relaxed);
    std::string outfile;
    run_metadata.step_stats().SerializeToString(&outfile);
    std::ofstream ofs("timeline-" + std::to_string(index));
    ofs << outfile;
    ofs.close();
  }

 private:
  void PrintDebugString() {
    LOG(INFO) << "tracing_: " << tracing_
              << "next_tracing_step_: " << next_tracing_step_
              << "interval_step_: " << interval_step_
              << "tracing_count_: " << tracing_count_
              << "limit_step_: " << limit_step_;
  }

 private:
  bool tracing_ = false;
  int64_t next_tracing_step_ = 0;
  int64_t interval_step_ = 1;
  int64_t tracing_count_ = 0;
  int64_t limit_step_ = 0;
  std::atomic<int64_t> curr_step_;

  mutex mu_;
};

} // namespace processor
} // namespace tensorflow

#endif // ODL_PROCESSOR_SERVING_TRACER_H

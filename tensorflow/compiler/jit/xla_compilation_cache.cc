/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"

#include <numeric>

#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

constexpr int64 XlaCompilationCache::kDefaultCompilationThreshold;

XlaCompilationCache::XlaCompilationCache(xla::LocalClient* client,
                                         DeviceType device_type)
    : client_(client), device_type_(std::move(device_type)), async_compilation_() {}

XlaCompilationCache::~XlaCompilationCache() {
  // Ensure any use of our programs have completed by waiting for all stream
  // executors to complete.
  for (auto* executor : client_->backend().stream_executors()) {
    bool ok = executor->SynchronizeAllActivity();
    if (!ok) {
      LOG(ERROR) << "Error synchronizing activity while waiting for all "
                    "programs to complete";
    }
  }
  // TODO(b/110813685): Think about the program ownership model. Programs are
  // currently owned by the compilation cache which means we must wait for
  // program completion in the destructor. There are multiple compilation caches
  // around, which complicates things a little. Perhaps having programs be
  // shared_ptrs (an invasive change) would make the model easier to reason
  // about?
}

string XlaCompilationCache::DebugString() const {
  return "XLA JIT compilation cache";
}

// Compute a string signature which encodes the shapes of the
// arguments in the supplied list.
string XlaCompilationCache::Signature::HumanString() const {
  string result = name;
  for (const auto& a : arg_shapes) {
    absl::StrAppend(&result, ",", DataTypeString(a.first));
    absl::StrAppend(&result, " [", absl::StrJoin(a.second, ","), "]");
  }

  for (const auto& v : arg_values) {
    absl::StrAppend(&result, "; ", v.DebugString());
  }
  return result;
}

bool XlaCompilationCache::Signature::operator==(const Signature& other) const {
  if (name != other.name) return false;
  if (arg_shapes != other.arg_shapes) return false;

  if (arg_values.size() != other.arg_values.size()) return false;
  for (int i = 0; i < arg_values.size(); ++i) {
    if (arg_values[i].dtype() != other.arg_values[i].dtype() ||
        arg_values[i].shape() != other.arg_values[i].shape() ||
        arg_values[i].tensor_data() != other.arg_values[i].tensor_data()) {
      return false;
    }
  }
  return true;
}

uint64 XlaCompilationCache::Signature::Hash::operator()(
    const XlaCompilationCache::Signature& signature) const {
  uint64 h = std::hash<string>()(signature.name);
  for (const auto& arg : signature.arg_shapes) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, std::hash<int>()(arg.second.size()));
    for (int dim : arg.second) {
      h = Hash64Combine(h, std::hash<int>()(dim));
    }
  }
  for (const auto& arg : signature.arg_values) {
    h = Hash64Combine(
        h, Hash64(arg.tensor_data().data(), arg.tensor_data().size()));
  }
  return h;
}

xla::StatusOr<XlaCompilationCache::Signature>
XlaCompilationCache::BuildSignature(
    const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args) {
  Signature signature;
  signature.name = Canonicalize(function.name(), AttrSlice(&function.attr()));
  for (const XlaCompiler::Argument& arg : args) {
    switch (arg.kind) {
      case XlaCompiler::Argument::kConstant:
        signature.arg_values.push_back(arg.constant_value);
        break;
      case XlaCompiler::Argument::kParameter:
      case XlaCompiler::Argument::kResource:
        signature.arg_shapes.emplace_back(arg.type, arg.DimensionSizes());
        break;
      default:
        return errors::InvalidArgument(
            "Unhandled argument kind in XlaCompilationCache: ",
            arg.HumanString());
    }
  }
  return std::move(signature);
}

Status XlaCompilationCache::BuildExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result,
    std::unique_ptr<xla::LocalExecutable>* executable) {
  VLOG(2) << "Compiling to local executable";

  std::vector<const xla::Shape*> argument_layouts(
      result.xla_input_shapes.size());
  for (int i = 0; i < result.xla_input_shapes.size(); ++i) {
    argument_layouts[i] = &result.xla_input_shapes[i];
  }
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(options.device_ordinal != -1
                                       ? options.device_ordinal
                                       : client_->default_device_ordinal());
  build_options.set_result_layout(result.xla_output_shape);
  build_options.set_device_allocator(options.device_allocator.get());

  auto compile_result =
      client_->Compile(*result.computation, argument_layouts, build_options);
  if (!compile_result.ok()) {
    return compile_result.status();
  }
  *executable = std::move(compile_result.ValueOrDie());
  return Status::OK();
}

Status XlaCompilationCache::Compile(
    const XlaCompiler::Options& options, const NameAttrList& function,
    std::vector<XlaCompiler::Argument>& args,
    const XlaCompiler::CompileOptions& compile_options,
    CompileMode compile_mode,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
  auto compile_fn = [=](XlaCompiler* compiler,
                        XlaCompiler::CompilationResult* result) {
    return compiler->CompileFunction(compile_options, function, args, result);
  };
  return CompileImpl(options, function, args, compile_fn, compile_mode,
                     out_compilation_result, out_executable);
}

static bool IsMegamorphic(
    int64 compile_count,
    int64 execution_count,
    uint64 max_compile_time_s) {
  const int64 kCompileThreshold = 10;
  const int64 kMinExecutionsPerCompile = 50;
  const uint64 kMaxCompileTimeThreshold = 30;

  // This heuristic is trying to capture the following property: have we sunk a
  // certain minimum amount of compile time into the cluster that didn't quite
  // "pay off"?
  return (compile_count > kCompileThreshold &&
          execution_count < kMinExecutionsPerCompile * compile_count) ||
          max_compile_time_s > kMaxCompileTimeThreshold;
}

Status XlaCompilationCache::CompileSingleOp(
    const XlaCompiler::Options& options,
    absl::Span<const XlaCompiler::Argument> args, OpKernelContext* ctx,
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
  const NodeDef& def = ctx->op_kernel().def();
  NameAttrList name;
  name.set_name(def.op());
  *name.mutable_attr() = def.attr();
  // Remove the "_class" attribute from the attribute set used to create the
  // compilation cache key. This attribute is information for the colocator
  // and causes false uniqueness between nodes.
  name.mutable_attr()->erase("_class");
  auto compile_op = [&](XlaCompiler* compiler,
                        XlaCompiler::CompilationResult* result) {
    std::vector<DataType> result_dtypes(ctx->num_outputs());
    for (int i = 0; i < result_dtypes.size(); ++i) {
      result_dtypes[i] = ctx->expected_output_dtype(i);
    }
    return compiler->CompileSingleOp(compile_options, ctx->op_kernel().def(),
                                     args, result_dtypes, result);
  };
  return CompileImpl(options, name, args, compile_op, CompileMode::kStrict,
                     out_compilation_result, out_executable);
}

namespace {
// Print something that users can search for to definitively ascertain that XLA
// was used for their TF model.
//
// Prints only once to avoid spamming LOG(INFO).
void LogOnceXlaCompiledFirstCluster() {
  static absl::once_flag log_once;
  absl::call_once(log_once, [] {
    LOG(INFO) << "Compiled cluster using XLA!  This line is logged at most "
                 "once for the lifetime of the process.";
  });
}
}  // namespace

Status XlaCompilationCache::CompileStrict(
    Entry* entry, const XlaCompiler::Options& options,
    const string &function_name,
    const std::function<Status(XlaCompiler* compiler,
                               XlaCompiler::CompilationResult*)>& compile_fn) {
  tensorflow::Env* env = tensorflow::Env::Default();
  const uint64 compile_start_us = env->NowMicros();

  XlaCompiler compiler(options);
  entry->compile_state = CompileState::kCompiled;

  entry->compilation_status =
      compile_fn(&compiler, &entry->compilation_result);
  TF_RETURN_IF_ERROR(entry->compilation_status);
  CHECK_EQ(entry->executable.get(), nullptr);
  entry->compilation_status =
      BuildExecutable(options, entry->compilation_result, &entry->executable);

  const uint64 compile_end_us = env->NowMicros();
  const uint64 compile_time_us = compile_end_us - compile_start_us;
  metrics::UpdateXlaCompilationTime(compile_time_us);
  {
    mutex_lock lock(cluster_compile_stats_mu_);
    auto it = cluster_compile_stats_.find(function_name);
    const uint64 compile_time_s = compile_time_us / 1.0e6;
    it->second.compile_count++;
    it->second.cumulative_compile_time_us += compile_time_us;
    it->second.max_compile_time_s = std::max(it->second.max_compile_time_s,
                                             compile_time_s);
    LogOnceXlaCompiledFirstCluster();
    VLOG(1) << "compiled " << function_name << " "
            << it->second.compile_count
            << " times, compile time: " << compile_time_us
            << " us, cumulative: " << it->second.cumulative_compile_time_us
            << " us ("
            << tensorflow::strings::HumanReadableElapsedTime(compile_time_s)
            << " / "
            << tensorflow::strings::HumanReadableElapsedTime(
                   it->second.cumulative_compile_time_us / 1.0e6)
            << ")";

    XlaJitCompilationActivity jit_compilation_activity;
    jit_compilation_activity.set_cluster_name(function_name);
    jit_compilation_activity.set_compile_count(it->second.compile_count);
    jit_compilation_activity.set_compile_time_us(compile_time_us);
    jit_compilation_activity.set_cumulative_compile_time_us(
        it->second.cumulative_compile_time_us);
    TF_RETURN_IF_ERROR(
        BroadcastXlaActivity(std::move(jit_compilation_activity)));
  }

  return Status::OK();
}

Status XlaCompilationCache::CompileAsynchronous(
    Entry* entry, const XlaCompiler::Options& options,
    const string &function_name,
    const std::function<Status(XlaCompiler* compiler,
                               XlaCompiler::CompilationResult*)>& compile_fn) {
  entry->compile_state = CompileState::kCompiling; // still under caller's lock.
  {
    mutex_lock lock(async_compilation_.async_compilation_mu_);
    async_compilation_.nrof_ongoing_compilations++;
  }
  // don't move the above code into the thread function!!!

  // passing options by value into the lamba increases the refcount on
  // options.device_allocator, keeping it alive for the duration of the
  // compilation
  async_compilation_.compiler_threads.Schedule([=] {
      Entry tmp;
      VLOG(2) << "Starting asynchronous compilation of cluster "
              << function_name << '.';
      (void)CompileStrict(&tmp, options, function_name, compile_fn);
      VLOG(2) << "Finished asynchronous compililation of cluster "
              << function_name << '.';
      {
        mutex_lock lock(async_compilation_.async_compilation_mu_);
        async_compilation_.nrof_ongoing_compilations--;
      }
      { // populate original entry with compilation result
        mutex_lock entry_lock(entry->mu);
        entry->compilation_result = tmp.compilation_result;
        entry->compile_state = tmp.compile_state;
        entry->compilation_status = tmp.compilation_status;
        entry->executable = std::move(tmp.executable);
      }
    }
  );
  return Status::OK();
}

Status XlaCompilationCache::CompileImpl(
    const XlaCompiler::Options& options, const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args,
    const std::function<Status(XlaCompiler* compiler,
                               XlaCompiler::CompilationResult*)>& compile_fn,
    CompileMode compile_mode,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
  DCHECK_NE(out_executable, nullptr);
  VLOG(2) << "XlaCompilationCache::Compile " << DebugString();

  VLOG(2) << "num_inputs=" << args.size();
  if (VLOG_IS_ON(3)) {
    for (int i = 0; i < args.size(); i++) {
      VLOG(3) << i << ": " << args[i].HumanString();
    }
  }
  absl::optional<int64> compile_threshold;
  if (compile_mode == CompileMode::kLazy) {
    compile_threshold = kDefaultCompilationThreshold;
  } else if (compile_mode == CompileMode::kAsync) {
    compile_threshold = 0; // for now, always compile right away
  }

  TF_ASSIGN_OR_RETURN(Signature signature, BuildSignature(function, args));
  string function_name = function.name();
  string human_signature = VLOG_IS_ON(3) ? signature.HumanString() : function_name;
  VLOG(2) << "Signature: " << human_signature;

  // The outer lock protects the existence of the cache entry. It does not
  // protect the contents of the cache entry.
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    // Find or create a cache entry.
    std::unique_ptr<Entry>& e = cache_[signature];
    if (!e) {
      e.reset(new Entry);
    }
    entry = e.get();
  }

  // We avoid compiling clusters that have "gone megamorphic" i.e. have an
  // excessive amount of shape dynamism.
  bool is_megamorphic;

  {
    mutex_lock lock(cluster_compile_stats_mu_);
    auto it =
        cluster_compile_stats_.emplace(function_name, ClusterCompileStats{})
            .first;

    it->second.execution_count++;
    // The is_megamorphic bit is "sticky".  We assume clusters that have been
    // observed to be megamorphic once stay megamorphic forever.
    it->second.is_megamorphic |=
        IsMegamorphic(/*compile_count=*/it->second.compile_count,
                      /*execution_count=*/it->second.execution_count,
                      /*max_compile_time_s=*/it->second.max_compile_time_s);
    is_megamorphic = it->second.is_megamorphic;
  }

  // Acquire the cache entry lock and compile, if necessary.
  // TODO(phawkins): this locking will need to be restructured when we implement
  // cache eviction.
  mutex_lock entry_lock(entry->mu);
  int64 current_request_count = ++entry->request_count;
  VLOG(2) << "Compilation cache entry hit: "
          << static_cast<int>(entry->compile_state)
          << " signature: " << human_signature << " with request count "
          << current_request_count << " and compile threshold "
          << compile_threshold.value_or(0);
  bool return_null = false;
  CompileState state = entry->compile_state;
  if (state == CompileState::kUncompiled) {
    const bool should_compile = [&] {
      if (compile_mode == CompileMode::kStrict) {
        // Lazy compilation is disabled.
        return true;
      }

      if (is_megamorphic) {
        BroadcastOptimizationRemark(XlaOptimizationRemark::MEGAMORPHIC_FUNCTION,
                                    function_name)
            .IgnoreError();
        VLOG(2) << "Not compiling cluster " << function_name
                << " because it is megamorphic.";
        return false;
      }

      if (compile_mode == CompileMode::kAsync) {
        // asynchronous compilation is enabled.
        {
          mutex_lock lock(async_compilation_.async_compilation_mu_);
          if (async_compilation_.nrof_ongoing_compilations >=
                async_compilation_.kMaxNrofOngoingCompilations) {
            VLOG(2) << "Not asynchronously compiling cluster " << function_name
                    << " because of too many ongoing compilations.";
            return false;
          }
        }
      }

      bool reached_compile_threshold =
          current_request_count >= *compile_threshold;
      if (!reached_compile_threshold) {
        VLOG(2)
            << "Not compiling cluster " << function_name
            << " because it has not reached compile threshold; threshold is "
            << *compile_threshold << " execution count "
            << current_request_count << ".";
      }
      return reached_compile_threshold;
    }();

    if (!should_compile) {
      VLOG(2) << "Not compiling for signature: " << human_signature;
      return_null = true;
    } else if (compile_mode == CompileMode::kAsync) {
      VLOG(2) << "Queueing asynchronous compilation for signature: " << human_signature;
      TF_RETURN_IF_ERROR(
        CompileAsynchronous(entry, options, function_name, compile_fn));
      return_null = true;
    } else {
      VLOG(2) << "Instantly compiling for signature: " << human_signature;
      TF_RETURN_IF_ERROR(
        CompileStrict(entry, options, function_name, compile_fn));
    }
  } else if (state == CompileState::kCompiling) {
      VLOG(2) << "Ongoing asynchronous compilation for signature: " << human_signature;
      return_null = true;
  } else if (state == CompileState::kCompiled) {
      VLOG(2) << "Already Compiled for signature: " << human_signature;
  }
  if (return_null) {
    *out_compilation_result = nullptr;
    *out_executable = nullptr;
  } else {
    TF_RETURN_IF_ERROR(entry->compilation_status);
    *out_compilation_result = &entry->compilation_result;
    *out_executable = entry->executable.get();
  }
  return Status::OK();
}

}  // namespace tensorflow

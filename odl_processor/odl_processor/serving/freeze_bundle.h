/*************************************************************************
    > File Name: freeze_bundle.h
    > Author: Yaozheng
    > Mail: yaozheng.wyz@alibaba-inc.com
    > Created Time: Thu Jan 18 06:28:13 2018
 ************************************************************************/
#ifndef TENSORFLOW_EAS_FREEZE_BUNDLE_H
#define TENSORFLOW_EAS_FREEZE_BUNDLE_H
#include <sys/stat.h>
#include <unistd.h>
#include <memory>
#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
class Session;
class GraphDef;
class SessionOptions;
class Status;
}  // namespace tensorflow

using tensorflow::Session;
using tensorflow::GraphDef;
using tensorflow::SessionOptions;
using tensorflow::Status;

namespace processor {

struct FreezeBundle {
  std::unique_ptr<Session> session;
  GraphDef graph_def;

  ~FreezeBundle() {
    if (session) {
      session->Close().IgnoreError();
    }
  }
  FreezeBundle() = default;
};

Status LoadFreezeModel(const SessionOptions& options,
                       const std::string& freeze_model_file,
                       FreezeBundle* bundle);

bool MaybeFreezeModelFile(const std::string& freeze_model_file);

}  // namespace processor

#endif

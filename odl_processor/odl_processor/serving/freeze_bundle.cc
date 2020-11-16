/*************************************************************************
    > File Name: freeze_bundle.cpp
    > Author: Yaozheng
    > Mail: yaozheng.wyz@alibaba-inc.com
    > Created Time: Thu Jan 18 06:27:17 2018
 ************************************************************************/
#include "freeze_bundle.h"

using tensorflow::Env;
using tensorflow::ReadBinaryProto;

namespace {

Status LoadGraphIntoSession(const GraphDef& graph_def,
                            const SessionOptions& session_options,
                            std::unique_ptr<Session>* session) {
  session->reset(NewSession(session_options));
  return (*session)->Create(graph_def);
}

}  // namespace

namespace processor {

Status LoadFreezeModel(const SessionOptions& options,
                       const std::string& freeze_model_file,
                       FreezeBundle* bundle) {
  TF_RETURN_IF_ERROR(
      ReadBinaryProto(Env::Default(), freeze_model_file, &(bundle->graph_def)));
  TF_RETURN_IF_ERROR(
      LoadGraphIntoSession(bundle->graph_def, options, &bundle->session));
  return Status::OK();
}

bool MaybeFreezeModelFile(const std::string& freeze_model_file) {
  struct stat file_stat;
  stat(freeze_model_file.c_str(), &file_stat);
  if (S_ISREG(file_stat.st_mode)) return true;
  return false;
}

}  // namespace processor 

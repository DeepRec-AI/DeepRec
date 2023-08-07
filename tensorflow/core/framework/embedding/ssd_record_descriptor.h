/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_RECORD_DESCRIPTOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_RECORD_DESCRIPTOR_H_

#include <map>
#include <vector>
#include <cstdlib>
#include <iomanip>

#include "tensorflow/core/framework/embedding/embedding_var_dump_iterator.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {

template <class K>
class SsdRecordDescriptor {
 public:
  //prefix of embedding file
  tstring file_prefix;
  //keys in ssd storage
  std::vector<K> key_list;
  //file ids of features
  std::vector<int64> key_file_id_list;
  //offsets in the file of features
  std::vector<int64> key_offset_list;
   //files in ssd storage
  std::vector<int64> file_list;
  //number of invalid records in the file
  std::vector<int64> invalid_record_count_list;
  //number of records in the file
  std::vector<int64> record_count_list;

  void GenerateCheckpoint(const std::string& prefix,
                          const std::string& var_name) {
    DumpSsdMeta(prefix, var_name);
    CopyEmbeddingFilesToCkptDir(prefix, var_name);
  }

 private:
  template<typename T>
  void DumpSection(const std::vector<T>& data_vec,
                   const std::string& section_str,
                   BundleWriter* writer,
                   std::vector<char>& dump_buffer) {
    EVVectorDataDumpIterator<T> iter(data_vec);
    SaveTensorWithFixedBuffer(
        section_str,
        writer, dump_buffer.data(),
        dump_buffer.size(), &iter,
        TensorShape({data_vec.size()}));
  }

  void DumpSsdMeta(const std::string& prefix,
                   const std::string& var_name) {
    std::fstream fs;
    std::string var_name_temp(var_name);
    std::string new_str = "_";
    int64 pos = var_name_temp.find("/");
    while (pos != std::string::npos) {
      var_name_temp.replace(pos, 1, new_str.data(), 1);
      pos = var_name_temp.find("/");
    }

    std::string ssd_record_path =
        prefix + "-" + var_name_temp + "-ssd_record";
    BundleWriter ssd_record_writer(Env::Default(),
                                   ssd_record_path);
    size_t bytes_limit = 8 << 20;
    std::vector<char> dump_buffer(bytes_limit);

    DumpSection(key_list, "keys",
                &ssd_record_writer, dump_buffer);
    DumpSection(key_file_id_list, "keys_file_id",
                &ssd_record_writer, dump_buffer);
    DumpSection(key_offset_list, "keys_offset",
                &ssd_record_writer, dump_buffer);
    DumpSection(file_list, "files",
                &ssd_record_writer, dump_buffer);
    DumpSection(invalid_record_count_list, "invalid_record_count",
                &ssd_record_writer, dump_buffer);
    DumpSection(record_count_list, "record_count",
                &ssd_record_writer, dump_buffer);

    ssd_record_writer.Finish();
  }

  void CopyEmbeddingFilesToCkptDir(
      const std::string& prefix,
      const std::string& var_name) {
    std::string var_name_temp(var_name);
    std::string new_str = "_";
    int64 pos = var_name_temp.find("/");
    while (pos != std::string::npos) {
      var_name_temp.replace(pos, 1, new_str.data(), 1);
      pos = var_name_temp.find("/");
    }

    std::string embedding_folder_path =
        prefix + "-" + var_name_temp + "-emb_files/";
    Status s = Env::Default()->CreateDir(embedding_folder_path);
    if (errors::IsAlreadyExists(s)) {
      int64 undeleted_files, undeleted_dirs;
      Env::Default()->
          DeleteRecursively(embedding_folder_path,
                            &undeleted_files,
                            &undeleted_dirs);
      Env::Default()->CreateDir(embedding_folder_path);
    }

    for (int64 i = 0; i < file_list.size(); i++) {
      int64 file_id = file_list[i];
      std::stringstream old_ss;
      old_ss << std::setw(4) << std::setfill('0') << file_id << ".emb";
      std::string file_path = file_prefix + old_ss.str();
      std::string file_name = file_path.substr(file_path.rfind("/"));
      std::stringstream new_ss;
      new_ss << file_id << ".emb";
      std::string new_file_path = embedding_folder_path + new_ss.str();
      Status s = Env::Default()->CopyFile(file_path, new_file_path);
      if (!s.ok()) {
        LOG(FATAL)<<"Copy file "<<file_path<<" failed!";
      }
    }
  }
};

}  // namespace embedding
}  // namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_RECORD_DESCRIPTOR_H_

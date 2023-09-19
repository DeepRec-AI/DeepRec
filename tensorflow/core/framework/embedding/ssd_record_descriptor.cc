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
#include "tensorflow/core/framework/embedding/ssd_record_descriptor.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"

namespace tensorflow {
namespace embedding {
template <class K>
template <class T>
void SsdRecordDescriptor<K>::DumpSection(
    const std::vector<T>& data_vec,
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
#define REGISTER_KERNELS(ktype, ttype)                   \
  template void SsdRecordDescriptor<ktype>::DumpSection( \
      const std::vector<ttype>&, const std::string&,       \
      BundleWriter*, std::vector<char>&);
REGISTER_KERNELS(int32, int32);
REGISTER_KERNELS(int32, int64);                                    
REGISTER_KERNELS(int64, int32);
REGISTER_KERNELS(int64, int64);
#undef REGISTER_KERNELS

template <class K>
void SsdRecordDescriptor<K>::DumpSsdMeta(
    const std::string& prefix,
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
#define REGISTER_KERNELS(ktype)                               \
  template void SsdRecordDescriptor<ktype>::DumpSsdMeta(  \
      const std::string&, const std::string&);
REGISTER_KERNELS(int32);                                    
REGISTER_KERNELS(int64);
#undef REGISTER_KERNELS
}//namespace embedding
}//namespace tensorflow

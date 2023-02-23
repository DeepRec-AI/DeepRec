#include <set>
#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/framework/versions.h"
#include "tensorflow/core/lib/gtl/stl_util.h"

#include <fstream>
#include "include/json/json.h"

char* kCheckpointPathPrefix = "checkpoint_path_prefix";
char* kOutputFile = "output_file";
char* kTensorRenameMap = "tensor_rename_map";
/*
 * Transform TFRA::de ckpt to DeepRec::ev ckpt
 *
 * TFRA::de
 *   de.get_variable('a', ..., devices=['ps:0', 'ps:1'])
 *   CKPT tensor:
 *     a-1of2-keys
 *     a-1of2-values
 *     a-2of2-keys
 *     a-2of2-values
 *
 * DeepRec::ev
 *   tf.get_embedding_variable('a', ..., partitioner=fixed_size_partitioner(2))
 *   CKPT tensor:
 *     a/part_0-keys
 *     a/part_0-values
 *     a/part_0-versions
 *     a/part_1-keys
 *     a/part_1-values
 *     a/part_1-versions
 */
namespace tensorflow {

class EVCkptTransformer {
 public:
  explicit EVCkptTransformer(const std::map<std::string, std::string>& prefixs)
  : prefixs_(prefixs) {

  }

  Status Transform(const std::string& src_ckpt, const std::string& dst_file) {
    BundleReader reader(Env::Default(), src_ckpt);
    auto s = reader.status();
    if (!s.ok()) {
      LOG(FATAL) << "Restore EV failure, create BundleReader error:"
                 << s.ToString();
    }

    reader.Seek(kHeaderEntryKey);

    for (reader.Next(); reader.Valid(); reader.Next()) {
      std::string key_string(reader.key());
      auto it = prefixs_.find(key_string);
      if (it != prefixs_.end()) {
        LOG(INFO) << "tensor name: " << key_string << " -> " << it->second;
        key_string = it->second;
      }

      BundleEntryProto* entry = &entries_[key_string];
      CHECK(entry->ParseFromArray(reader.value().data(), reader.value().size()));
    }
    return Flush(dst_file);
  }

 private:
  Status Flush(const std::string& dst_file) {
    std::unique_ptr<WritableFile> file;
    Status s = Env::Default()->NewWritableFile(dst_file, &file);
    if (!s.ok()) return s;
    {
      // N.B.: the default use of Snappy compression may not be supported on all
      // platforms (e.g. Android).  The metadata file is small, so this is fine.
      table::Options options;
      options.compression = table::kNoCompression;
      table::TableBuilder builder(options, file.get());
      // Header entry.
      BundleHeaderProto header;
      header.set_num_shards(1);
      header.set_endianness(BundleHeaderProto::LITTLE);
      if (!port::kLittleEndian) header.set_endianness(BundleHeaderProto::BIG);
      VersionDef* version = header.mutable_version();
      version->set_producer(kTensorBundleVersion);
      version->set_min_consumer(kTensorBundleMinConsumer);

      builder.Add(kHeaderEntryKey, header.SerializeAsString());

      // All others.
      for (const auto& p : entries_) {
        builder.Add(p.first, p.second.SerializeAsString());
      }
      s = builder.Finish();
    }
    s.Update(file->Close());
    return Status::OK();
  }

 private:
  std::map<std::string, std::string> prefixs_;
  std::map<string, BundleEntryProto> entries_;

};

} // tensorflow namespace

int main (int argc, char** argv) {
  if (2 != argc) {
    LOG(ERROR) << "Usage: ./ev_ckpt_transformer config.json";
    return -2;
  }
  std::ifstream f(argv[1]);
  Json::Value config;
  f >> config;
  std::map<std::string, std::string> rename_map;
  for (const auto& key : config[kTensorRenameMap].getMemberNames()) {
    Json::Value value = config[kTensorRenameMap][key];
    rename_map[key] = value.asString();
  }
  tensorflow::EVCkptTransformer transformer(rename_map);
  std::string src = config[kCheckpointPathPrefix].asString();
  std::string dst = config[kOutputFile].asString();
  LOG(INFO) << "transforming checkpoint from " << src << " to " << dst << " ...";
  tensorflow::Status s = transformer.Transform(src, dst);
  LOG(INFO) << "transform done. " << s.ToString();
  return 0;
}

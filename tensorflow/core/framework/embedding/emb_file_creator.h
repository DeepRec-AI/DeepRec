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
======================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMB_FILE_CREATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMB_FILE_CREATOR_H_
#include <string>
#include <map>

#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/framework/embedding/emb_file.h"

namespace tensorflow {
namespace embedding {

enum class IoScheme {
  MMAP_AND_MADVISE = 0,
  MMAP = 1,
  DIRECT_IO = 2,
  INVALID = 3
};

class EmbFileCreator {
 public:
  virtual EmbFile* Create(const std::string& path,
                          const size_t version,
                          const size_t buffer_size) = 0;
};

class MmapAndMadviseEmbFileCreator : public EmbFileCreator {
 public:
  EmbFile* Create(const std::string& path,
                  const size_t version,
                  const size_t buffer_size) override {
    return new MmapMadviseEmbFile(path, version, buffer_size);
  }
};

class MmapEmbFileCreator : public EmbFileCreator {
 public:
  EmbFile* Create(const std::string& path,
                  const size_t version,
                  const size_t buffer_size) override {
    return new MmapEmbFile(path, version, buffer_size);
  }
};

class DirectIoEmbFileCreator : public EmbFileCreator {
 public:
  EmbFile* Create(const std::string& path,
                  const size_t version,
                  const size_t buffer_size) override {
    return new DirectIoEmbFile(path, version, buffer_size);
  }
};


class EmbFileCreatorFactory {
 public:
  static EmbFileCreator* Create(const std::string& io_scheme) {
    std::map<std::string, IoScheme> scheme_map{
      {"mmap_and_madvise", IoScheme::MMAP_AND_MADVISE},
      {"mmap", IoScheme::MMAP},
      {"directio", IoScheme::DIRECT_IO}
    };
    
    IoScheme scheme = IoScheme::INVALID;
    if (scheme_map.find(io_scheme) != scheme_map.end()) {
      scheme = scheme_map[io_scheme];
    }

    switch (scheme) {
      case IoScheme::MMAP_AND_MADVISE:
        static MmapAndMadviseEmbFileCreator mmap_madvise_file_creator;
        return &mmap_madvise_file_creator;
      case IoScheme::MMAP:
        static MmapEmbFileCreator mmap_file_creator;
        return &mmap_file_creator;
      case IoScheme::DIRECT_IO:
        static DirectIoEmbFileCreator directio_file_creator;
        return &directio_file_creator;
      default:
        LOG(WARNING)<<"Invalid IO scheme of SSDHASH,"
                    <<" use default mmap_and_advise scheme.";
        static MmapAndMadviseEmbFileCreator default_file_creator;
        return &default_file_creator;
    }
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMB_FILE_CREATOR_H_

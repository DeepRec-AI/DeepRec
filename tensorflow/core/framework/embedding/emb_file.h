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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMB_FILE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMB_FILE_H_
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <map>
#include <malloc.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>

#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {
class EmbFile {
 public:
  EmbFile(const std::string& path, size_t ver, int64 buffer_size)
    :version_(ver),
     file_size_(buffer_size),
     count_(0),
     invalid_count_(0),
     is_deleted_(false) {
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << ver << ".emb";
    filepath_ = path + ss.str();
    OpenFstream();
  }

  virtual ~EmbFile() {}
  virtual void Reopen() = 0;
  virtual void Read(char* val, const size_t val_len,
      const size_t offset) = 0;

  virtual void DeleteFile() {
    is_deleted_ = true;
    if (fs_.is_open()) {
      fs_.close();
    }
    close(fd_);
    std::remove(filepath_.c_str());
  }

  void LoadExistFile(const std::string& old_file_path,
                     size_t count, size_t invalid_count) {
    Env::Default()->CopyFile(old_file_path, filepath_);
    Reopen();
    count_ = count;
    invalid_count_ = invalid_count;
  }

  void Flush() {
    if (fs_.is_open()) {
      fs_.flush();
    }
  }

  void MapForRead() {
    file_addr_for_read_ = (char*)mmap(nullptr, file_size_, PROT_READ,
        MAP_PRIVATE, fd_, 0);
  }

  void UnmapForRead() {
    munmap((void*)file_addr_for_read_, file_size_);
  }

  void ReadWithMemcpy(char* val, const size_t val_len,
      const size_t offset) {
    memcpy(val, file_addr_for_read_ + offset, val_len);
  }

  void Write(const char* val, const size_t val_len) {
    if (fs_.is_open()) {
      fs_.write(val, val_len);
      posix_fadvise(fd_, 0, file_size_, POSIX_FADV_DONTNEED);
    } else {
      fs_.open(filepath_,
          std::ios::app | std::ios::in | std::ios::out |
          std::ios::binary);
      fs_.write(val, val_len);
      fs_.close();
    }
  }

  size_t Count() const {
    return count_;
  }

  void AddCount(size_t n) {
    count_ += n;
  }

  size_t InvalidCount() const {
    return invalid_count_;
  }

  void AddInvalidCount(size_t n) {
    invalid_count_ += n;
  }

  void AddInvalidCountAtomic(size_t n) {
    __sync_fetch_and_add(&invalid_count_, n);
  }

  size_t Version() const {
    return version_;
  }

  bool IsDeleted() const {
    return is_deleted_;
  }

  bool IsNeedToBeCompacted() {
    return (count_ >= invalid_count_) && (count_ / 3 < invalid_count_);
  }

 protected:
  void OpenFstream() {
    fs_.open(filepath_,
             std::ios::app |
             std::ios::in  |
             std::ios::out |
             std::ios::binary);
    CHECK(fs_.good());
  }
  void CloseFstream() {
    if (fs_.is_open()) {
      fs_.close();
    }
  }

 private:
  size_t version_;
  size_t count_;
  size_t invalid_count_;
  char* file_addr_for_read_;
  std::fstream fs_;

 protected:
  int64 file_size_;
  int fd_;
  bool is_deleted_;
  std::string filepath_;
};

class MmapMadviseEmbFile : public EmbFile {
 public: 
  MmapMadviseEmbFile(const std::string& path,
                     size_t ver,
                     int64 buffer_size)
    :EmbFile(path, ver, buffer_size) {
    EmbFile::fd_ = open(EmbFile::filepath_.data(), O_RDONLY);
    file_addr_ = (char*)mmap(nullptr, EmbFile::file_size_, PROT_READ,
        MAP_PRIVATE, fd_, 0);
  }

  void Reopen() override {
    CloseFstream();
    munmap((void*)file_addr_, EmbFile::file_size_);
    close(EmbFile::fd_);
    OpenFstream();
    EmbFile::fd_ = open(EmbFile::filepath_.data(), O_RDONLY);
    file_addr_ = (char*)mmap(nullptr, EmbFile::file_size_, PROT_READ,
        MAP_PRIVATE, fd_, 0);
  }

  void DeleteFile() override {
    is_deleted_ = true;
    CloseFstream();
    munmap((void*)file_addr_, EmbFile::file_size_);
    close(EmbFile::fd_);
    std::remove(EmbFile::filepath_.c_str());
  }

  void Read(char* val, const size_t val_len,
            const size_t offset) override {
    memcpy(val, file_addr_ + offset, val_len);
    int err = madvise(file_addr_, EmbFile::file_size_, MADV_DONTNEED);
    if (err < 0) {
      LOG(FATAL)<<"Failed to madvise the page, file_addr_: "
                <<(void*)file_addr_<<", file_size: "
                <<EmbFile::file_size_;
    }
  }

 private:
  char* file_addr_;
};

class MmapEmbFile : public EmbFile {
 public: 
  MmapEmbFile(const std::string& path,
              size_t ver,
              int64 buffer_size)
    :EmbFile(path, ver, buffer_size) {
    EmbFile::fd_ = open(EmbFile::filepath_.data(), O_RDONLY);
  }

  void Reopen() override {
    CloseFstream();
    close(EmbFile::fd_);
    OpenFstream();
    EmbFile::fd_ = open(EmbFile::filepath_.data(), O_RDONLY);
  }

  void Read(char* val, const size_t val_len,
            const size_t offset) override {
    char* file_addr_tmp =
          (char*)mmap(nullptr, EmbFile::file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    memcpy(val, file_addr_tmp + offset, val_len);
    munmap((void*)file_addr_tmp, EmbFile::file_size_);
  }
};

class DirectIoEmbFile : public EmbFile {
 public:
  DirectIoEmbFile(const std::string& path,
                  size_t ver,
                  int64 buffer_size)
    :EmbFile(path, ver, buffer_size) {
    EmbFile::fd_ = open(EmbFile::filepath_.data(), O_RDONLY|O_DIRECT);
  }

  void Reopen() override {
    EmbFile::CloseFstream();
    close(EmbFile::fd_);
    OpenFstream();
    EmbFile::fd_ = open(EmbFile::filepath_.data(), O_RDONLY|O_DIRECT);
  }

  void Read(char* val, const size_t val_len,
            const size_t offset) override {
    size_t page_size = getpagesize();
    int pages_to_read = val_len / page_size;
    if (val_len % page_size != 0) {
      pages_to_read += 1;
    }
    if (offset + val_len >= page_size * pages_to_read) {
      pages_to_read += 1;
    }
    int aligned_offset = offset - (offset % page_size);
    char* read_buffer = (char*)memalign(page_size, page_size * pages_to_read);

    int status = pread(EmbFile::fd_,
                       (void*)read_buffer,
                       page_size * pages_to_read,
                       aligned_offset);
    if (status < 0) {
      LOG(FATAL)<<"Failed to pread, read size: "
                <<page_size * pages_to_read
                <<", offset: "<<aligned_offset;
    }
    memcpy(val, read_buffer + (offset % page_size), val_len);
    free(read_buffer);
  }
};

} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMB_FILE_H_

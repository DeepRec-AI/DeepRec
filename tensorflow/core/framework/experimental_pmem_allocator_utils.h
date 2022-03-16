#ifndef TENSORFLOW_CORE_FRAMEWORK_EXPERIMENTAL_PMEM_ALLOCATOR_UTILS_H_
#define TENSORFLOW_CORE_FRAMEWORK_EXPERIMENTAL_PMEM_ALLOCATOR_UTILS_H_

#include <atomic>
#include <memory>
#include <string.h>
#include <sys/stat.h>
#include <unordered_set>

#include "tensorflow/core/lib/core/spin_lock.h"

namespace tensorflow {

template <typename T>
class NoCopyArray {
 public:
  template <typename... A>
  explicit NoCopyArray(uint64_t size, A&&... args) : size_(size) {
    data_ = (T*)malloc(sizeof(T) * size);
    for (uint64_t i = 0; i < size; i++) {
      new (data_ + i) T(std::forward<A>(args)...);
    }
  }

  NoCopyArray(const NoCopyArray<T>& v) = delete;
  NoCopyArray& operator=(const NoCopyArray&) = delete;
  NoCopyArray(NoCopyArray&&) = delete;

  NoCopyArray() : size_(0), data_(nullptr){};

  ~NoCopyArray() {
    if (data_ != nullptr) {
      for (uint64_t i = 0; i < size_; i++) {
        data_[i].~T();
      }
      free(data_);
    }
  }

  T& back() {
    assert(size_ > 0);
    return data_[size_ - 1];
  }

  T& front() {
    assert(size_ > 0);
    return data_[0];
  }

  T& operator[](uint64_t index) {
    if (index >= size_) {
      std::abort();
    }
    return data_[index];
  }

  uint64_t size() { return size_; }

 private:
  T* data_;
  uint64_t size_;
};

class ThreadManager;

struct AllocatorThread {
 public:
  AllocatorThread() : id(-1), thread_manager(nullptr) {}

  ~AllocatorThread();

  void Release();

  int id;
  std::shared_ptr<ThreadManager> thread_manager;
};

class ThreadManager : public std::enable_shared_from_this<ThreadManager> {
 public:
  ThreadManager(uint32_t max_threads) : ids_(0), max_threads_(max_threads) {}

  int MaybeInitThread(AllocatorThread& t);

  void Release(const AllocatorThread& t);

 private:
  std::atomic<uint32_t> ids_;
  std::unordered_set<uint32_t> usable_id_;
  uint32_t max_threads_;
  spin_lock spin_;
};

inline int create_dir_if_missing(const std::string& name) {
  int res = mkdir(name.c_str(), 0755) != 0;
  if (res != 0) {
    if (errno != EEXIST) {
      return res;
    } else {
      struct stat s;
      if (stat(name.c_str(), &s) == 0) {
        return S_ISDIR(s.st_mode) ? 0 : res;
      }
    }
  }
  return res;
}

}  // namespace tensorflow

#endif

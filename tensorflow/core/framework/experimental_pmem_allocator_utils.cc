#include <assert.h>

#include "tensorflow/core/framework/experimental_pmem_allocator.h"
#include "tensorflow/core/lib/core/spin_lock.h"
namespace tensorflow {

void AllocatorThread::Release() {
  assert(id == -1 || thread_manager != nullptr);
  if (thread_manager) {
    thread_manager->Release(*this);
    thread_manager = nullptr;
  }
  id = -1;
}

AllocatorThread::~AllocatorThread() { Release(); }

int ThreadManager::MaybeInitThread(AllocatorThread& t) {
  if (t.id < 0) {
    if (!usable_id_.empty()) {
      std::lock_guard<spin_lock> lg(spin_);
      if (!usable_id_.empty()) {
        auto it = usable_id_.begin();
        t.id = *it;
        usable_id_.erase(it);
        t.thread_manager = shared_from_this();
        return t.id;
      }
    }
    int id = ids_.fetch_add(1, std::memory_order_relaxed);
    if (id >= max_threads_) {
      return -1;
    }
    t.id = id;
    t.thread_manager = shared_from_this();
  }
  return t.id;
}

void ThreadManager::Release(const AllocatorThread& t) {
  std::lock_guard<spin_lock> lg(spin_);
  if (t.id >= 0) {
    usable_id_.insert(t.id);
  }
}
}  // namespace tensorflow

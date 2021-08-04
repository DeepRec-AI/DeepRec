#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VALUE_PTR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VALUE_PTR_H_

#include <pthread.h>
#include <bitset>
#include <atomic>
#include <memory>

#include "tensorflow/core/framework/typed_allocator.h"

namespace tensorflow {

template <class V>
class ValuePtr {
 public:
  ValuePtr(size_t size) {
    /*___________________________________________________________________________________________________
      |           |               |             |               |    embedding     |       slot       |
      | number of | each bit a V* | global step | freq counter  |        V*        |        V*        |
      | embedding |    1 valid    |             |               | actually pointer | actually pointer |...
      |  columns  |   0 no-valid  |    int64    |     int64     |    by alloctor   |    by alloctor   |
      |  (8 bits) |   (56 bits)   |  (8 bytes)  |   (8 bytes)   |     (8 bytes)    |     (8 bytes)    |
      ---------------------------------------------------------------------------------------------------
     */
    // we make sure that we store at least one embedding column
    ptr_ = (void*) malloc(sizeof(int64) * (3 + size));
    memset(ptr_, 0, sizeof(int64) * (3 + size));
  }

  ~ValuePtr() {
    free(ptr_);
  }

  V* GetOrAllocate(Allocator* allocator, int64 value_len, const V* default_v, int emb_index) {
    // fetch meta
    unsigned long metaorig = ((unsigned long*)ptr_)[0];
    unsigned int embnum = metaorig & 0xff;
    std::bitset<56> metadata(metaorig >> 8);
     
    if (!metadata.test(emb_index)) {

      while(flag_.test_and_set(std::memory_order_acquire)); 
      
      if (metadata.test(emb_index)) 
        return ((V**)((int64*)ptr_ + 3))[emb_index];
      // need to realloc
      /*
      if (emb_index + 1 > embnum) {
        ptr_ = (void*)realloc(ptr_, sizeof(int64) * (1 + emb_index + 1));
      }*/
      embnum++ ;
      int64 alloc_value_len = value_len;
      //if (allocate_version) {
      //  alloc_value_len = value_len + (sizeof(int64) + sizeof(V) - 1) / sizeof(V);
      //}
      V* tensor_val = TypedAllocator::Allocate<V>(allocator, alloc_value_len, AllocationAttributes());

      memcpy(tensor_val, default_v, sizeof(V) * value_len);

      ((V**)((int64*)ptr_ + 3))[emb_index]  = tensor_val;

      metadata.set(emb_index);
      // NOTE:if we use ((unsigned long*)((char*)ptr_ + 1))[0] = metadata.to_ulong();
      // the ptr_ will be occaionally  modified from 0x7f18700912a0 to 0x700912a0
      // must use  ((V**)ptr_ + 1 + 1)[emb_index] = tensor_val;  to avoid
      ((unsigned long*)(ptr_))[0] = (metadata.to_ulong() << 8) | embnum;

      flag_.clear(std::memory_order_release);
      return tensor_val;
    } else {
      return ((V**)((int64*)ptr_ + 3))[emb_index];
    }
  }


  // simple getter for V* and version
  V* GetValue(int emb_index) {
    unsigned long metaorig = ((unsigned long*)ptr_)[0];
    std::bitset<56> metadata(metaorig >> 8);
    if (metadata.test(emb_index)) {
      return ((V**)((int64*)ptr_ + 3))[emb_index];
    } else {
      return nullptr;
    }
  }

  int64 GetStep() {
    return *((int64*)ptr_ + 1);
  }

  void SetStep(int64 gs) {
    *((int64*)ptr_ + 1) = gs;
  }

  int64 GetFreq() {
    return *((int64*)ptr_ + 2);
  }

  void AddFreq() {
    while(flag_.test_and_set(std::memory_order_acquire)); 
    *((int64*)ptr_ + 2) = std::min(max_freq_, *((int64*)ptr_ + 2) + 1);
    flag_.clear(std::memory_order_release);
  }
  	
  void Destroy(int64 value_len) {
    unsigned long metaorig = ((unsigned long*)ptr_)[0];
    unsigned int embnum = metaorig & 0xff;
    std::bitset<56> metadata(metaorig >> 8);

    for (int i = 0; i< embnum; i++) {
      if (metadata.test(i)) {
        V* val = ((V**)((int64*)ptr_ + 3))[i];
        if (val != nullptr) {
          TypedAllocator::Deallocate(cpu_allocator(), val, value_len);
        }
      }
    }
  }

 private:
  void* ptr_;
  static const int64 max_freq_ = 100000;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VALUE_PTR_H_

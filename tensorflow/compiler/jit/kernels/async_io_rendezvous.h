/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_JIT_ASYNC_IO_RENDEZVOUS_
#define TENSORFLOW_COMPILER_JIT_ASYNC_IO_RENDEZVOUS_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace tensorflow {

// Mimics the Rendezvous class in Tensorflow but tailors it for the use of
// AsyncIO.
class AsyncIoRendezvous {
 public:
  // Returns a key for rendezvous concatenating device name and tensor name.
  static string GetRendezvousKey(const string& device,
                                 const string& tensor_name) {
    return strings::StrCat(device, ";", tensor_name);
  }

  static uint64 GetRendezvousKeyHash(StringPiece k) {
    return Hash64(k.data(), k.size());
  }

 public:
  // TensorPayload stores either Tensorflow::Tensor or XlaTensor. XlaTensor is
  // represented by an se::DeviceMemoryBase `addr` and an xla::Shape `shape`.
  // If `addr` is null, `tensor` is valid and being used. Otherwise, `addr` and
  // `shape` are valid.
  struct TensorPayload {
    Tensor tensor;
    // `addr` and `shape` represent XlaTensor.
    se::DeviceMemoryBase addr;
    xla::Shape shape;
  };

  typedef std::function<void(const Status&, const TensorPayload&)> DoneCallback;

  explicit AsyncIoRendezvous() {}

  Status Send(uint64 key_hash, const TensorPayload& val);

  void RecvAsync(uint64 key_hash, DoneCallback done);

  // Initializes the MutexedItemQueue indexed by the key. Returns the hashed
  // key. To avoid a global lock for Send/Recv operations, Initializing queues
  // is required before Send/Recv operations.
  void InitializeRendezvousQueue(const uint64 key_hash) {
    mutex_lock l(mu_);
    MutexedItemQueue& mq = table_[key_hash];
    CHECK(mq.queue.empty());
  }

  void FinalizeRendezvousQueue(const uint64 key_hash) {
    mutex_lock l(mu_);
    MutexedItemQueue& mq = table_[key_hash];
    CHECK(mq.queue.empty());
    table_.erase(key_hash);
  }

 private:
  struct Item {
    DoneCallback waiter = nullptr;
    TensorPayload value;

    // Returns true iff this item represents a value being sent.
    bool IsSendValue() const { return this->waiter == nullptr; }
  };

  // By invariant, the item queue under each key is of the form
  //   [item.IsSendValue()]* meaning each item is a sent message.
  // or
  //   [!item.IsSendValue()]* meaning each item is a waiter.
  typedef std::deque<Item*> ItemQueue;
  struct MutexedItemQueue {
    ItemQueue queue GUARDED_BY(mu);
    mutex mu;  // mutex for per-queue operation.
  };
  typedef absl::flat_hash_map<uint64, MutexedItemQueue> Table;

  // This global lock `mu_` only guards `table_` but does not guard per-queue
  // operations.
  mutex mu_;
  Table table_ GUARDED_BY(mu_);

  ~AsyncIoRendezvous() {}

  TF_DISALLOW_COPY_AND_ASSIGN(AsyncIoRendezvous);
};

AsyncIoRendezvous* GetXlaAsyncIORendezvous();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_ASYNC_IO_RENDEZVOUS_

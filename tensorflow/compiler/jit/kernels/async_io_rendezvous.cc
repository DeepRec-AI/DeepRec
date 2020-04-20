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

#include "tensorflow/compiler/jit/kernels/async_io_rendezvous.h"

namespace tensorflow {

Status AsyncIoRendezvous::Send(uint64 key_hash, const TensorPayload& val) {
  VLOG(2) << "Send " << this << " " << key_hash;

  MutexedItemQueue& mq = table_[key_hash];
  ItemQueue* queue = &mq.queue;
  mutex& mu = mq.mu;
  mu.lock();
  if (queue->empty() || queue->front()->IsSendValue()) {
    // There is no waiter for this message. Append the message
    // into the queue. The waiter will pick it up when arrives.
    // Only send-related fields need to be filled.
    VLOG(2) << "Enqueue Send Item (key hash:" << key_hash << ", queue:" << queue
            << "). ";
    Item* item = new Item;
    item->value = val;
    queue->push_back(item);
    mu.unlock();
    return Status::OK();
  }

  VLOG(2) << "Consume Recv Item (key hash:" << key_hash << "). ";
  // There is an earliest waiter to consume this message.
  Item* item = queue->front();
  queue->pop_front();
  mu.unlock();

  // Notify the waiter by invoking its done closure, outside the
  // lock.
  CHECK(!item->IsSendValue());
  item->waiter(Status::OK(), val);
  delete item;
  return Status::OK();
}

void AsyncIoRendezvous::RecvAsync(uint64 key_hash, DoneCallback done) {
  VLOG(2) << "Recv " << this << " " << key_hash;

  MutexedItemQueue& mq = table_[key_hash];
  ItemQueue* queue = &mq.queue;
  mutex& mu = mq.mu;
  mu.lock();
  if (queue->empty() || !queue->front()->IsSendValue()) {
    // There is no message to pick up.
    // Only recv-related fields need to be filled.
    VLOG(2) << "Enqueue Recv Item (key hash:" << key_hash << "). ";
    Item* item = new Item;
    item->waiter = std::move(done);

    queue->push_back(item);
    mu.unlock();
    return;
  }

  VLOG(2) << "Consume Send Item (key hash:" << key_hash << "). ";
  // A message has already arrived and is queued in the table under
  // this key.  Consumes the message and invokes the done closure.
  Item* item = queue->front();
  queue->pop_front();
  mu.unlock();

  // Invokes the done() by invoking its done closure, outside scope
  // of the table lock.
  CHECK(item->IsSendValue());
  done(Status::OK(), item->value);
  delete item;
}

/*static*/ AsyncIoRendezvous* GetXlaAsyncIORendezvous() {
  static AsyncIoRendezvous* self = new AsyncIoRendezvous();
  return self;
}

}  // namespace tensorflow

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <list>
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace embedding {

template <class K>
class BatchCache {
public:
  BatchCache() {}
  virtual size_t get_evic_ids(K* evic_ids, size_t k_size) = 0;
  virtual void add_to_rank(const K* batch_ids, size_t batch_size) = 0;
  virtual size_t size() = 0;
};

template <class K>
class LRUCache : public BatchCache<K> {
private:
  class LRUNode {
  public:
    K id;
    LRUNode *pre, *next;
    LRUNode(K id) : id(id), pre(nullptr), next(nullptr) {}
  };
  LRUNode *head, *tail;
  std::map<K, LRUNode *> mp;
  mutex mu_;

public:
  LRUCache() {
    mp.clear();
    head = new LRUNode(0);
    tail = new LRUNode(0);
    head->next = tail;
    tail->pre = head;
  }

  size_t size() {
    mutex_lock l(mu_);
    return mp.size();
  }

  size_t get_evic_ids(K* evic_ids, size_t k_size) {
    mutex_lock l(mu_);
    size_t true_size = 0;
    LRUNode *evic_node = tail->pre;
    LRUNode *rm_node = evic_node;
    for (size_t i = 0; i < k_size && evic_node != head; ++i) {
      evic_ids[i] = evic_node->id;
      rm_node = evic_node;
      evic_node = evic_node->pre;
      mp.erase(rm_node->id);
      delete rm_node;
      true_size++;
    }
    evic_node->next = tail;
    tail->pre = evic_node;
    return true_size;
  }

  void add_to_rank(const K* batch_ids, size_t batch_size) {
    mutex_lock l(mu_);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      typename std::map<K, LRUNode *>::iterator it = mp.find(id);
      if (it != mp.end()) {
        LRUNode *node = it->second;
        node->pre->next = node->next;
        node->next->pre = node->pre;
        head->next->pre = node;
        node->next = head->next;
        head->next = node;
        node->pre = head;
      } else {
        LRUNode *newNode = new LRUNode(id);
        head->next->pre = newNode;
        newNode->next = head->next;
        head->next = newNode;
        newNode->pre = head;
        mp[id] = newNode;
      }
    }
  }
};

template <class K>
class LFUCache : public BatchCache<K> {
private:
  class LFUNode {
  public:
    K key;
    size_t freq;
    LFUNode(K key, size_t freq) : key(key), freq(freq) {}
  };
  size_t min_freq;
  size_t max_freq;
  std::unordered_map<K, typename std::list<LFUNode>::iterator> key_table;
  std::unordered_map<K, typename std::list<LFUNode>> freq_table;
  mutex mu_;

public:
  LFUCache() {
    min_freq = 0;
    max_freq = 0;
    key_table.clear();
    freq_table.clear();
  }

  size_t size() {
    mutex_lock l(mu_);
    return key_table.size();
  }

  size_t get_evic_ids(K *evic_ids, size_t k_size) {
    mutex_lock l(mu_);
    size_t true_size = 0;
    for (size_t i = 0; i < k_size; ++i) {
      auto rm_it = freq_table[min_freq].back();
      key_table.erase(rm_it.key);
      evic_ids[i] = rm_it.key;
      ++true_size;
      freq_table[min_freq].pop_back();
      if (freq_table[min_freq].size() == 0) {
        freq_table.erase(min_freq);
        ++min_freq;
        while (min_freq <= max_freq) {
          auto it = freq_table.find(min_freq);
          if (it == freq_table.end() || it->second.size() == 0) {
            ++min_freq;
          } else {
            break;
          }
        }
      }
    }
    return true_size;
  }

  void add_to_rank(const K *batch_ids, size_t batch_size) {
    mutex_lock l(mu_);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it = key_table.find(id);
      if (it == key_table.end()) {
        freq_table[1].push_front(LFUNode(id, 1));
        key_table[id] = freq_table[1].begin();
        min_freq = 1;
      } else {
        typename std::list<LFUNode>::iterator node = it->second;
        size_t freq = node->freq;
        freq_table[freq].erase(node);
        if (freq_table[freq].size() == 0) {
          freq_table.erase(freq);
          if (min_freq == freq)
            min_freq += 1;
        }
        max_freq = std::max(max_freq, freq + 1);
        freq_table[freq + 1].push_front(LFUNode(id, freq + 1));
        key_table[id] = freq_table[freq + 1].begin();
      }
    }
  }
};

} // embedding
} // tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <fstream>
#include <iomanip>
#include <map>
#include <queue>
#include <vector>

#include "sparsehash/dense_hash_map_lockless"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"

namespace tensorflow {

template <class V>
class ValuePtr;

namespace embedding {
template <class K, class V>
class SSDKV : public KVInterface<K, V> {
 public:
  explicit SSDKV(std::string path) {
    path_ = io::JoinPath(
        path, "ssd_kv_" + std::to_string(Env::Default()->NowMicros()) + "_");
    hash_map.max_load_factor(0.8);
    hash_map.set_empty_key_and_value(-1, nullptr);
    hash_map.set_counternum(1);
    hash_map.set_deleted_key(-2);
    current_version = 0;
    current_offset = 0;
    buffer_size = 1 << 27;  // Write 128MB at once.
    buffer_cur = 0;
    open_file_count = 100;
    EmbFile* ef = new EmbFile(path_, current_version, buffer_size);
    emb_files.emplace_back(ef);
    total_app_count = 0;
    new_value_ptr_fn_ = [](size_t size) {
      return new NormalContiguousValuePtr<V>(ev_allocator(), size);
    };
    compaction_ration = 2;
  }

  void SetTotalDims(int total_dims) {
    total_dims_ = total_dims;
    val_len = sizeof(FixedLengthHeader) + total_dims_ * sizeof(V);
    max_app_count = buffer_size / val_len;
    write_buffer = new char[buffer_size];
    unsigned int max_key_count = 1 + int(buffer_size / val_len);
    key_buffer = new K[max_key_count];
  }

  ~SSDKV() {
    if (buffer_cur > 0) {
      emb_files[current_version]->Write(write_buffer, buffer_cur * val_len);
      TF_CHECK_OK(UpdateFlushStatus());
      buffer_cur = 0;
    }
    for (auto it : emb_files) {
      if (!it->is_deleted) {
        it->DeleteFile();
      }
    }
    delete[] write_buffer;
    delete[] key_buffer;
  }

  Status UpdateFlushStatus() {
    for (int i = 0; i < buffer_cur; ++i) {
      auto iter = hash_map.find_wait_free(key_buffer[i]);
      if (iter.first == EMPTY_KEY_) {
        return errors::NotFound("Unable to find Key: ", key_buffer[i],
                                " in SSDKV.");
      } else {
        iter.second->flushed = true;
      }
    }
    return Status::OK();
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) {
    auto iter = hash_map.find_wait_free(key);
    if (iter.first == EMPTY_KEY_) {
      return errors::NotFound("Unable to find Key: ", key, " in SSDKV.");
    } else {
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      EmbPosition* posi = iter.second;
      if (posi->flushed) {
        emb_files[posi->version]->Read((char*)(val->GetPtr()), val_len,
                                       posi->offset);
      } else {
        memcpy((char*)val->GetPtr(), write_buffer + posi->buffer_offset,
               val_len);
      }
      *value_ptr = val;
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) { return Status::OK(); }

  Status BatchInsert(std::vector<K> keys,
                     std::vector<ValuePtr<V>*> value_ptrs) {
    return BatchCommit(keys, value_ptrs);
  }

  Status BatchCommit(std::vector<K> keys,
                     std::vector<ValuePtr<V>*> value_ptrs) {
    SingleThreadDynamicCompaction();
    total_app_count += keys.size();
    for (int i = 0; i < keys.size(); i++) {
      CheckBuffer();
      SaveKV(keys[i], value_ptrs[i]);
      delete value_ptrs[i];
    }
    return Status::OK();
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) {
    SingleThreadDynamicCompaction();
    total_app_count++;
    CheckBuffer();
    SaveKV(key, value_ptr);
    return Status::OK();
  }

  Status CommitForRestore(K key, ValuePtr<V>* value_ptr) {
    SingleThreadDynamicCompaction();
    total_app_count++;
    CheckBuffer();
    SaveKVForRestore(key, value_ptr);
    return Status::OK();
  }

  Status Remove(K key) {
    if (hash_map.erase_lockless(key)) {
      return Status::OK();
    } else {
      return errors::NotFound("Unable to find Key: ", key, " in SSDKV.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>*>* value_ptr_list) {
    for (auto it : hash_map) {
      key_list->emplace_back(it.first);
      EmbPosition* posi = it.second;
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      if (posi->flushed) {
        emb_files[posi->version]->Read((char*)(val->GetPtr()), val_len,
                                       posi->offset);
      } else {
        memcpy((char*)val->GetPtr(), write_buffer + posi->buffer_offset,
               val_len);
      }
      value_ptr_list->emplace_back(val);
    }
    return Status::OK();
  }

  int64 Size() const { return hash_map.size(); }

  void FreeValuePtr(ValuePtr<V>* value_ptr) { delete value_ptr; }

 private:
  void CheckBuffer() {
    if (buffer_cur * val_len + val_len > buffer_size) {
      emb_files[current_version]->Write(write_buffer, buffer_cur * val_len);
      emb_files[current_version]->app_count += buffer_cur;
      emb_files[current_version]->Flush();
      if (emb_files[current_version]->app_count >= max_app_count) {
        ++current_version;
        current_offset = 0;
        emb_files.emplace_back(new EmbFile(path_, current_version, buffer_size));
      }
      TF_CHECK_OK(UpdateFlushStatus());
      buffer_cur = 0;
    }
  }

  void SaveKV(K key, const ValuePtr<V>* value_ptr, bool is_compaction = false) {
    EmbPosition* ep = new EmbPosition(current_offset, current_version,
                                      buffer_cur * val_len, false);

    current_offset += val_len;
    memcpy(write_buffer + buffer_cur * val_len, (char*)value_ptr->GetPtr(),
           val_len);
    key_buffer[buffer_cur] = key;
    ++buffer_cur;

    auto iter = hash_map.insert_lockless(std::move(
        std::pair<K, EmbPosition*>(key, const_cast<EmbPosition*>(ep))));
    if ((*(iter.first)).second != ep) {
      if (!is_compaction)
        evict_file_set.insert((*(iter.first)).second->version);
      EmbPosition* old_posi = (*(iter.first)).second;
      __sync_bool_compare_and_swap(&((*(iter.first)).second),
                                   (*(iter.first)).second, ep);
      if(pos_out_of_date.size() > 100000){
        EmbPosition* posi = pos_out_of_date.front();
        delete posi;
        pos_out_of_date.erase(pos_out_of_date.begin());
      }
      pos_out_of_date.emplace_back(old_posi);
    }
  }

  void SaveKVForRestore(K key, ValuePtr<V>* value_ptr,
                        bool is_compaction = false) {
    EmbPosition* ep = new EmbPosition(current_offset, current_version,
                                      buffer_cur * val_len, false);
    auto iter = hash_map.insert_lockless(std::move(
        std::pair<K, EmbPosition*>(key, const_cast<EmbPosition*>(ep))));

    if ((*(iter.first)).second != ep) {
      if (!is_compaction)
        evict_file_set.insert((*(iter.first)).second->version);
      EmbPosition* posi = (*(iter.first)).second;
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      if (posi->flushed) {
        emb_files[posi->version]->Read((char*)(val->GetPtr()), val_len,
                                       posi->offset);
      } else {
        memcpy((char*)val->GetPtr(), write_buffer + posi->buffer_offset,
               val_len);
      }
      value_ptr->MergeValuePtr(val, total_dims_);
      __sync_bool_compare_and_swap(&((*(iter.first)).second),
                                   (*(iter.first)).second, ep);
      delete val;
      delete posi;
    }

    current_offset += val_len;
    memcpy(write_buffer + buffer_cur * val_len, (char*)value_ptr->GetPtr(),
           val_len);
    key_buffer[buffer_cur] = key;
    ++buffer_cur;
  }

  void SingleThreadDynamicCompaction() {
    uint64 start, end;
    int64 hash_size = hash_map.size_lockless();
    if (hash_size * 3 / 2 < total_app_count ||
        total_app_count - hash_size > 10000000) {
      // delete the evict_files
      for (auto it : evict_file_map) {
        emb_files[it.first]->DeleteFile();
      }

      // flush the data in buffer
      evict_file_map.clear();
      emb_files[current_version]->Write(write_buffer, buffer_cur * val_len);
      emb_files[current_version]->app_count += buffer_cur;
      emb_files[current_version]->Flush();
      TF_CHECK_OK(UpdateFlushStatus());
      size_t save_version = current_version;
      ++current_version;
      current_offset = 0;
      emb_files.emplace_back(new EmbFile(path_, current_version, buffer_size));

      buffer_cur = 0;
      total_app_count = hash_size;  // important
      // Initialize evict_file_map
      for (auto it : evict_file_set) {
        std::vector<std::pair<K, EmbPosition*>> tmp;
        evict_file_map[it] = tmp;
      }
      evict_file_set.clear();
      for (auto it : hash_map) {
        EmbPosition* posi = it.second;
        auto iter = evict_file_map.find(posi->version);
        if (iter != evict_file_map.end()) {
          (*iter).second.emplace_back(it);
        }
      }
      // read embeddings and write to new file
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      for (auto it : evict_file_map) {
        EmbFile* file = emb_files[it.first];
        for (auto it_vec : it.second) {
          EmbPosition* posi = it_vec.second;
          file->Read((char*)(val->GetPtr()), val_len, posi->offset);
          CheckBuffer();
          SaveKV(it_vec.first, val, true);
        }
      }
      delete val;
    }
  }
  std::string DebugString() const {
    LOG(INFO) << "map info size:" << Size();
    LOG(INFO) << "map info bucket_count:" << hash_map.bucket_count();
    LOG(INFO) << "map info load_factor:" << hash_map.load_factor();
    LOG(INFO) << "map info max_load_factor:" << hash_map.max_load_factor();
    LOG(INFO) << "map info min_load_factor:" << hash_map.min_load_factor();
    return "";
  }

 private:
  size_t val_len;
  char* write_buffer;
  K* key_buffer;
  size_t buffer_size;
  size_t buffer_cur;
  size_t total_app_count;
  std::string path_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;
  int total_dims_;
  int16 open_file_count;

  class EmbPosition {
   public:
    size_t offset;
    size_t version;
    size_t buffer_offset;
    bool flushed;
    EmbPosition(size_t o, size_t v, size_t bo, bool f)
        : offset(o), version(v), buffer_offset(bo), flushed(f) {}
    EmbPosition()
        : offset(-1), version(-1), buffer_offset(-1), flushed(false) {}
    void Print() {
      LOG(INFO) << "EmbPosition: "
                << "offset= " << offset << ", version= " << version
                << ", buffer_offset= " << buffer_offset
                << ", flushed= " << flushed;
    }
  };

  class EmbFile {
   public:
    std::fstream fs;
    size_t app_count;
    size_t version;
    std::string filepath;
    int fd;
    char* file_addr;
    int64 file_size;
    bool is_deleted;

    EmbFile(std::string path_, size_t ver, int64 buffer_size) {
      version = ver;
      file_size = buffer_size;
      std::stringstream ss;
      ss << std::setw(4) << std::setfill('0') << ver << ".emb";
      filepath = path_ + ss.str();
      fs.open(filepath,
              std::ios::app | std::ios::in | std::ios::out | std::ios::binary);
      fd = open(filepath.data(), O_RDONLY);
      file_addr =
          (char*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
      CHECK(fs.good());
      app_count = 0;
      is_deleted = false;
    }

    void DeleteFile() {
      if (fs.is_open()) fs.close();
      munmap((void*)file_addr, file_size);
      close(fd);
      std::remove(filepath.c_str());
      is_deleted = true;
    }

    void Flush() {
      if (fs.is_open()) {
        fs.flush();
      }
    }

    void Write(const char* val, const size_t val_len) {
      if (fs.is_open()) {
        fs.write(val, val_len);
      } else {
        fs.open(filepath, std::ios::app | std::ios::in | std::ios::out |
                              std::ios::binary);
        fs.write(val, val_len);
        fs.close();
      }
    }

    void Read(char* val, const size_t val_len, const size_t offset) {
      memcpy(val, file_addr + offset, val_len);
    }
  };

  float compaction_ration;
  size_t max_app_count;
  typedef google::dense_hash_map_lockless<K, EmbPosition*> LockLessHashMap;
  static const int EMPTY_KEY_ = -1;
  static const int DELETED_KEY_ = -2;
  LockLessHashMap hash_map;
  std::vector<EmbFile*> emb_files;
  EmbFile* active_file;
  std::vector<EmbFile*> files_out_of_date;
  std::vector<EmbPosition*> pos_out_of_date;
  std::set<int64> evict_file_set;
  std::map<int64, std::vector<std::pair<K, EmbPosition*>>> evict_file_map;
  size_t current_version;
  size_t current_offset;
};

}  // namespace embedding
}  // namespace tensorflow

#endif TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_

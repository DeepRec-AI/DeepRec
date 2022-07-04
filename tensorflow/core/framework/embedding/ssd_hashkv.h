#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASHKV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASHKV_H_

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
class EmbPosition {
 public:
  EmbPosition(int o, size_t v, int bo, bool f)
      : offset(o), version(v), buffer_offset(bo), flushed(f), invalid(false) {}
  EmbPosition()
      : offset(-1),
        version(-1),
        buffer_offset(-1),
        flushed(false),
        invalid(false) {}
  void Print() {
    LOG(INFO) << "EmbPosition: "
              << "offset= " << offset << ", version= " << version
              << ", buffer_offset= " << buffer_offset
              << ", flushed= " << flushed;
  }
 public:
  int offset;
  int buffer_offset;
  size_t version;
  bool flushed;
  bool invalid;
};

class EmbFile {
 public:
  EmbFile(const std::string& path_, size_t ver, int64 buffer_size)
  : version(ver),
    file_size(buffer_size),
    app_count(0),
    app_invalid_count(0),
    is_deleted(false) {
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << ver << ".emb";
    filepath = path_ + ss.str();
    fs.open(filepath,
            std::ios::app | std::ios::in | std::ios::out | std::ios::binary);
    fd = open(filepath.data(), O_RDONLY);
    CHECK(fs.good());
  }

  void DeleteFile() {
    is_deleted = true;
    if (fs.is_open()) fs.close();
    close(fd);
    std::remove(filepath.c_str());
  }

  void Flush() {
    if (fs.is_open()) {
      fs.flush();
    }
  }

  void Map() {
    file_addr = (char*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  }

  void Unmap() { munmap((void*)file_addr, file_size); }

  void ReadWithoutMap(char* val, const size_t val_len, const size_t offset) {
    memcpy(val, file_addr + offset, val_len);
  }

  void Write(const char* val, const size_t val_len) {
    if (fs.is_open()) {
      fs.write(val, val_len);
    } else {
      fs.open(filepath,
              std::ios::app | std::ios::in | std::ios::out | std::ios::binary);
      fs.write(val, val_len);
      fs.close();
    }
  }

  void Read(char* val, const size_t val_len, const size_t offset) {
    char* file_addr_tmp =
        (char*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    memcpy(val, file_addr_tmp + offset, val_len);
    munmap((void*)file_addr_tmp, file_size);
  }
 public:
  size_t app_count;
  size_t app_invalid_count;
  size_t version;
  int64 file_size;
  int fd;
  char* file_addr;
  bool is_deleted;
  std::string filepath;
  std::fstream fs;
};

template <class K>
class SSDIterator : public Iterator {
 public:
  SSDIterator(google::dense_hash_map_lockless<K, EmbPosition*>* hash_map,
              const std::vector<EmbFile*>& emb_files, int64 value_len,
              char* write_buffer)
      : emb_files_(emb_files),
        curr_file_(0),
        curr_vec_(0),
        value_len_(value_len),
        write_buffer_(write_buffer) {
    for (auto it : *hash_map) {
      EmbPosition* posi = it.second;
      if (!posi->invalid) {
        auto iter = file_map_.find(posi->version);
        if (iter == file_map_.end()) {
          std::vector<std::pair<K, EmbPosition*>> tmp;
          file_map_[posi->version] = tmp;
          file_id_vec_.emplace_back(posi->version);
        }
        file_map_[posi->version].emplace_back(it);
      }
    }
  }

  virtual ~SSDIterator() {}
  virtual bool Valid() { return !(curr_file_ == file_id_vec_.size()); }
  virtual void SeekToFirst() {
    curr_file_ = 0;
    curr_vec_ = 0;
    if (file_id_vec_.size() > 0) {
      int64 f_id = file_id_vec_[curr_file_];
      emb_files_[f_id]->Map();
    }
  }
  virtual void Next() {
    curr_vec_++;
    int64 f_id = file_id_vec_[curr_file_];
    if (curr_vec_ == file_map_[f_id].size()) {
      emb_files_[f_id]->Unmap();
      curr_vec_ = 0;
      curr_file_++;
      if (curr_file_ < file_id_vec_.size())
        emb_files_[file_id_vec_[curr_file_]]->Map();
    }
  }
  virtual void Key(char* val, int64 dim) {
    int64 f_id = file_id_vec_[curr_file_];
    memcpy((char*)val, &((file_map_[f_id])[curr_vec_].first), dim);
  }
  virtual void Value(char* val, int64 dim, int64 value_offset) {
    int64 f_id = file_id_vec_[curr_file_];
    EmbPosition* posi = (file_map_[f_id])[curr_vec_].second;
    if (posi->flushed) {
      emb_files_[posi->version]->
          ReadWithoutMap(val, dim,
                           posi->offset + value_offset + sizeof(FixedLengthHeader));
    } else {
      memcpy(val,
            write_buffer_ + posi->buffer_offset +
              value_offset + + sizeof(FixedLengthHeader),
            dim);
    }
  }

 private:
  int64 value_len_;
  int64 curr_file_;
  int64 curr_vec_;
  char* write_buffer_;
  std::map<int64, std::vector<std::pair<K, EmbPosition*>>> file_map_;
  std::vector<int64> file_id_vec_;
  std::vector<EmbFile*> emb_files_;
};

template <class K, class V>
class SSDHashKV : public KVInterface<K, V> {
 public:
  explicit SSDHashKV(const std::string& path, Allocator* alloc_)
  : current_version(0),
    current_offset(0),
    buffer_cur(0),
    alloc(alloc_),
    total_app_count(0) {
    path_ = io::JoinPath(
        path, "ssd_kv_" + std::to_string(Env::Default()->NowMicros()) + "_");
    hash_map.max_load_factor(0.8);
    hash_map.set_empty_key_and_value(-1, nullptr);
    hash_map.set_counternum(1);
    hash_map.set_deleted_key(-2);
    EmbFile* ef = new EmbFile(path_, current_version, buffer_size);
    emb_files.emplace_back(ef);
    new_value_ptr_fn_ = [this](size_t size) {
      return new NormalContiguousValuePtr<V>(alloc, size);
    };
  }

  void SetTotalDims(int total_dims) {
    total_dims_ = total_dims;
    val_len = sizeof(FixedLengthHeader) + total_dims_ * sizeof(V);
    max_app_count = buffer_size / val_len;
    write_buffer = new char[buffer_size];
    unsigned int max_key_count = 1 + int(buffer_size / val_len);
    key_buffer = new K[max_key_count];
  }

  Iterator* GetIterator() {
    return new SSDIterator<K>(&hash_map, emb_files, val_len, write_buffer);
  }

  ~SSDHashKV() {
    if (buffer_cur > 0) {
      emb_files[current_version]->Write(write_buffer, buffer_cur * val_len);
      TF_CHECK_OK(UpdateFlushStatus());
      buffer_cur = 0;
    }
    for (auto it : emb_files) {
      if (!it->is_deleted) {
        it->DeleteFile();
      }
      delete it;
    }
    delete[] write_buffer;
    delete[] key_buffer;
  }

  Status UpdateFlushStatus() {
    for (int i = 0; i < buffer_cur; ++i) {
      auto iter = hash_map.find_wait_free(key_buffer[i]);
      if (iter.first == EMPTY_KEY_) {
        return errors::NotFound("Unable to find Key: ", key_buffer[i],
                                " in SSDHashKV.");
      } else {
        iter.second->flushed = true;
      }
    }
    return Status::OK();
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) {
    auto iter = hash_map.find_wait_free(key);
    if (iter.first == EMPTY_KEY_) {
      return errors::NotFound("Unable to find Key: ", key, " in SSDHashKV.");
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
      posi->invalid = true;
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) { return Status::OK(); }

  Status BatchInsert(std::vector<K>& keys,
                     std::vector<ValuePtr<V>*>& value_ptrs) {
    return BatchCommit(keys, value_ptrs);
  }

  Status BatchCommit(std::vector<K>& keys,
                     std::vector<ValuePtr<V>*>& value_ptrs) {
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

  Status Remove(K key) {
    if (hash_map.erase_lockless(key)) {
      return Status::OK();
    } else {
      return errors::NotFound("Unable to find Key: ", key, " in SSDHashKV.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>*>* value_ptr_list) {
    return Status::OK();
  }

  int64 Size() const { return hash_map.size(); }

  void FreeValuePtr(ValuePtr<V>* value_ptr) { delete value_ptr; }

 private:
  void CheckBuffer() {
    size_t curr_buffer_offset = buffer_cur * val_len;
    if (curr_buffer_offset + val_len > buffer_size) {
      emb_files[current_version]->Write(write_buffer, curr_buffer_offset);
      emb_files[current_version]->app_count += buffer_cur;
      emb_files[current_version]->Flush();
      if (emb_files[current_version]->app_count >= max_app_count) {
        ++current_version;
        current_offset = 0;
        emb_files.emplace_back(
            new EmbFile(path_, current_version, buffer_size));
      }
      TF_CHECK_OK(UpdateFlushStatus());
      buffer_cur = 0;
    }
  }

  void SaveKV(K key, const ValuePtr<V>* value_ptr, bool is_compaction = false) {
    size_t curr_buffer_offset = buffer_cur * val_len;
    EmbPosition* ep = new EmbPosition(current_offset, current_version,
                                      curr_buffer_offset, false);

    current_offset += val_len;
    memcpy(write_buffer + curr_buffer_offset, (char*)value_ptr->GetPtr(),
           val_len);
    key_buffer[buffer_cur] = key;
    ++buffer_cur;

    auto iter = hash_map.insert_lockless(std::move(
        std::pair<K, EmbPosition*>(key, const_cast<EmbPosition*>(ep))));
    if ((*(iter.first)).second != ep) {
      int version = (*(iter.first)).second->version;
      if (!is_compaction) {
        emb_files[version]->app_invalid_count++;
        //A parameter that can be adjusted in the future
        if (version != current_version && emb_files[version]->app_count >=
                emb_files[version]->app_invalid_count &&
            emb_files[version]->app_count / 3 <
                emb_files[version]->app_invalid_count)
          evict_file_set.insert((*(iter.first)).second->version);
      }
      EmbPosition* old_posi = (*(iter.first)).second;
      __sync_bool_compare_and_swap(&((*(iter.first)).second),
                                   (*(iter.first)).second, ep);
      //A parameter that can be adjusted in the future
      if (pos_out_of_date.size() > cap_invalid_pos) {
        EmbPosition* posi = pos_out_of_date.front();
        delete posi;
        pos_out_of_date.pop_front();
      }
      pos_out_of_date.emplace_back(old_posi);
    }
  }

  void SingleThreadDynamicCompaction() {
    uint64 start, end;
    int64 hash_size = hash_map.size_lockless();
    //These parameter that can be adjusted in the future
    if (hash_size * 3 / 2 < total_app_count ||
        total_app_count - hash_size > cap_invalid_id) {
      // delete the evict_files
      for (auto it : evict_file_map) {
        emb_files[it.first]->DeleteFile();
      }
      // flush the data in buffer
      evict_file_map.clear();
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
        total_app_count -= file->app_invalid_count;
        file->Map();
        for (auto it_vec : it.second) {
          EmbPosition* posi = it_vec.second;
          file->ReadWithoutMap((char*)(val->GetPtr()), val_len, posi->offset);
          CheckBuffer();
          SaveKV(it_vec.first, val, true);
        }
        file->Unmap();
      }
      delete val;
    }
  }
  std::string DebugString() const {
    return strings::StrCat("map info size:", Size(),
                          ", map info bucket_count:",
                           hash_map.load_factor(),
                           ",map info load_factor:",
                           hash_map.load_factor(),
                           ", map info max_load_factor:",
                           hash_map.max_load_factor(),
                           ", map info min_load_factor: ",
                           hash_map.min_load_factor());
  }

 private:
  size_t val_len;
  size_t current_version;
  size_t current_offset;
  size_t buffer_cur;
  size_t total_app_count;
  size_t max_app_count;

  char* write_buffer;
  K* key_buffer;
  EmbFile* active_file;
  Allocator* alloc;

  int total_dims_;
  std::string path_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;

  typedef google::dense_hash_map_lockless<K, EmbPosition*> LockLessHashMap;
  LockLessHashMap hash_map;
  static const int EMPTY_KEY_;
  static const int DELETED_KEY_;
  static const int cap_invalid_pos;
  static const int cap_invalid_id;
  static const size_t buffer_size;


  std::vector<EmbFile*> emb_files;
  std::vector<EmbFile*> files_out_of_date;
  std::deque<EmbPosition*> pos_out_of_date;
  std::set<int64> evict_file_set;
  std::map<int64, std::vector<std::pair<K, EmbPosition*>>> evict_file_map;
};
template <class K, class V>
const int SSDHashKV<K, V>::EMPTY_KEY_ = -1;
template <class K, class V>
const int SSDHashKV<K, V>::DELETED_KEY_ = -2;
template <class K, class V>
const int SSDHashKV<K, V>::cap_invalid_pos = 100000;
template <class K, class V>
const int SSDHashKV<K, V>::cap_invalid_id = 10000000;
template <class K, class V>
const size_t SSDHashKV<K, V>::buffer_size = 1<<27;

}  // namespace embedding
}  // namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASHKV_H_

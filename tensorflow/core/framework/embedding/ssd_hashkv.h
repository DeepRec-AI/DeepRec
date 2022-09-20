#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASHKV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASHKV_H_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <cstdlib>

#include "sparsehash/dense_hash_map_lockless"
#include "sparsehash/dense_hash_set_lockless"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

template <class V>
class ValuePtr;

namespace embedding {
class EmbPosition {
 public:
  EmbPosition(int o, size_t v, int bo, bool f)
      : offset_(o), version_(v), buffer_offset_(bo), flushed_(f),
        invalid_(false) {}

  EmbPosition()
      : offset_(-1),
        version_(-1),
        buffer_offset_(-1),
        flushed_(false),
        invalid_(false) {}

  void Print() {
    LOG(INFO) << "EmbPosition: "
              << "offset = " << offset_
              << ", version = " << version_
              << ", buffer_offset = " << buffer_offset_
              << ", flushed = " << flushed_;
  }
 public:
  int offset_;
  int buffer_offset_;
  size_t version_;
  bool flushed_;
  bool invalid_;
};

class EmbFile {
 public:
  EmbFile(const std::string& path, size_t ver, int64 buffer_size)
  : version_(ver),
    file_size_(buffer_size),
    app_count_(0),
    app_invalid_count(0),
    is_deleted_(false) {
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << ver << ".emb";
    filepath_ = path + ss.str();
    fs_.open(filepath_,
            std::ios::app | std::ios::in | std::ios::out | std::ios::binary);
    fd_ = open(filepath_.data(), O_RDONLY);
    CHECK(fs_.good());
  }

  void DeleteFile() {
    is_deleted_ = true;
    if (fs_.is_open()) {
      fs_.close();
    }
    close(fd_);
    std::remove(filepath_.c_str());
  }

  void Flush() {
    if (fs_.is_open()) {
      fs_.flush();
    }
  }

  void Map() {
    file_addr_ = (char*)mmap(nullptr, file_size_, PROT_READ,
        MAP_PRIVATE, fd_, 0);
  }

  void Unmap() {
    munmap((void*)file_addr_, file_size_);
  }

  void ReadWithoutMap(char* val, const size_t val_len,
      const size_t offset) {
    memcpy(val, file_addr_ + offset, val_len);
  }

  void Write(const char* val, const size_t val_len) {
    if (fs_.is_open()) {
      fs_.write(val, val_len);
    } else {
      fs_.open(filepath_,
          std::ios::app | std::ios::in | std::ios::out |
          std::ios::binary);
      fs_.write(val, val_len);
      fs_.close();
    }
  }

  void Read(char* val, const size_t val_len, const size_t offset) {
    char* file_addr_tmp =
        (char*)mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    memcpy(val, file_addr_tmp + offset, val_len);
    munmap((void*)file_addr_tmp, file_size_);
  }

 public:
  size_t app_count_;
  size_t app_invalid_count;
  size_t version_;
  int64 file_size_;
  int fd_;
  char* file_addr_;
  bool is_deleted_;
  std::string filepath_;
  std::fstream fs_;
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
      if (!posi->invalid_) {
        auto iter = file_map_.find(posi->version_);
        if (iter == file_map_.end()) {
          std::vector<std::pair<K, EmbPosition*>> tmp;
          file_map_[posi->version_] = tmp;
          file_id_vec_.emplace_back(posi->version_);
        }
        file_map_[posi->version_].emplace_back(it);
      }
    }
  }

  virtual ~SSDIterator() {}

  virtual bool Valid() {
    return !(curr_file_ == file_id_vec_.size());
  }

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
    if (posi->flushed_) {
      emb_files_[posi->version_]->
          ReadWithoutMap(val, dim,
              posi->offset_ + value_offset + sizeof(FixedLengthHeader));
    } else {
      memcpy(val, write_buffer_ + posi->buffer_offset_ +
          value_offset + + sizeof(FixedLengthHeader), dim);
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
  explicit SSDHashKV(const std::string& path, Allocator* alloc)
  : current_version_(0),
    evict_version_(0),
    compaction_version_(0),
    current_offset_(0),
    buffer_cur_(0),
    alloc_(alloc),
    total_app_count_(0),
    val_len_(-1),
    compaction_thread_(nullptr) {
    path_ = io::JoinPath(
        path, "ssd_kv_" + std::to_string(Env::Default()->NowMicros()) + "_");
    hash_map_.max_load_factor(0.8);
    hash_map_.set_empty_key_and_value(-1, nullptr);
    hash_map_.set_counternum(16);
    hash_map_.set_deleted_key(-2);
    evict_file_set_.max_load_factor(0.8);
    evict_file_set_.set_empty_key_and_value(-1, -1);
    evict_file_set_.set_counternum(16);
    evict_file_set_.set_deleted_key(-2);
    EmbFile* ef = new EmbFile(path_, current_version_, BUFFER_SIZE);
    emb_files_.emplace_back(ef);
    new_value_ptr_fn_ = [this](size_t size) {
      return new NormalContiguousValuePtr<V>(alloc_, size);
    };
    is_async_compaction_ = true;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_SSDHASH_ASYNC_COMPACTION", true,
          &is_async_compaction_));
    if (!is_async_compaction_) {
      LOG(INFO) <<
        "Use Sync Compactor in SSDHashKV of Multi-tier Embedding Stroage!";
      compaction_fn_ = [this](){Compaction();}; 
      check_buffer_fn_ = [this](){CheckBuffer();};
      save_kv_fn_ = [this](K key, const ValuePtr<V>* value_ptr,
          bool is_compaction=false) {
        SaveKV(key, value_ptr, is_compaction);
      };
    } else {
      LOG(INFO) <<
        "Use Async Compactor in SSDHashKV of Multi-tier Embedding Stroage!";
      compaction_fn_ = [](){};
      check_buffer_fn_ = [this](){CheckBufferAsync();};
      save_kv_fn_ = [this](K key, const ValuePtr<V>* value_ptr,
          bool is_compaction=false) {
        SaveKVAsync(key, value_ptr, is_compaction);
      };
      compaction_thread_ = Env::Default()->StartThread(
          ThreadOptions(), "COMPACTION", [this]() {
            CompactionThread();
          });
    }
  }

  void SetTotalDims(int total_dims) {
    total_dims_ = total_dims;
    val_len_ = sizeof(FixedLengthHeader) + total_dims_ * sizeof(V);
    max_app_count_ = BUFFER_SIZE / val_len_;
    write_buffer_ = new char[BUFFER_SIZE];
    unsigned int max_key_count = 1 + int(BUFFER_SIZE / val_len_);
    key_buffer_ = new K[max_key_count];
    done_ = true;
  }

  Iterator* GetIterator() {
    return new SSDIterator<K>(&hash_map_, emb_files_, val_len_,
        write_buffer_);
  }

  ~SSDHashKV() {
    if (buffer_cur_ > 0) {
      if (!is_async_compaction_) {
        emb_files_[current_version_]->Write(write_buffer_,
            buffer_cur_ * val_len_);
      } else {
        emb_files_[evict_version_]->Write(write_buffer_,
            buffer_cur_ * val_len_);
      }
      TF_CHECK_OK(UpdateFlushStatus());
      buffer_cur_ = 0;
    }
    for (auto it : emb_files_) {
      if (!it->is_deleted_) {
        it->DeleteFile();
      }
      delete it;
    }
    delete[] write_buffer_;
    delete[] key_buffer_;
    if (compaction_thread_) {
      shutdown_cv_.notify_all();
      shutdown_ = true;
      delete compaction_thread_;
    }
  }

  Status UpdateFlushStatus() {
    for (int i = 0; i < buffer_cur_; ++i) {
      auto iter = hash_map_.find_wait_free(key_buffer_[i]);
      if (iter.first == EMPTY_KEY) {
        return errors::NotFound("Unable to find Key: ",
            key_buffer_[i], " in SSDHashKV.");
      } else {
        iter.second->flushed_ = true;
      }
    }
    return Status::OK();
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == EMPTY_KEY) {
      return errors::NotFound("Unable to find Key: ", key, " in SSDHashKV.");
    } else {
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      EmbPosition* posi = iter.second;
      if (posi->flushed_) {
        emb_files_[posi->version_]->Read((char*)(val->GetPtr()),
            val_len_, posi->offset_);
      } else {
        memcpy((char*)val->GetPtr(),
            write_buffer_ + posi->buffer_offset_, val_len_);
      }
      *value_ptr = val;
      posi->invalid_ = true;
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) {
    return Status::OK();
  }

  Status BatchInsert(const std::vector<K>& keys,
                     const std::vector<ValuePtr<V>*>& value_ptrs) {
    return BatchCommit(keys, value_ptrs);
  }

  Status BatchCommit(const std::vector<K>& keys,
                     const std::vector<ValuePtr<V>*>& value_ptrs) {
    compaction_fn_();
    __sync_fetch_and_add(&total_app_count_, keys.size());
    for (int i = 0; i < keys.size(); i++) {
      check_buffer_fn_();
      save_kv_fn_(keys[i], value_ptrs[i], false);
      delete value_ptrs[i];
    }
    return Status::OK();
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) {
    compaction_fn_();
    __sync_fetch_and_add(&total_app_count_, 1);
    check_buffer_fn_();
    save_kv_fn_(key, value_ptr, false);
    return Status::OK();
  }

  Status Remove(K key) {
    if (hash_map_.erase_lockless(key)) {
      return Status::OK();
    } else {
      return errors::NotFound("Unable to find Key: ",
          key, " in SSDHashKV.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>*>* value_ptr_list) {
    return Status::OK();
  }

  int64 Size() const {
    return hash_map_.size();
  }

  void FreeValuePtr(ValuePtr<V>* value_ptr) {
    delete value_ptr;
  }

 private:
  void WriteFile(size_t version, size_t curr_buffer_offset) {
    emb_files_[version]->Write(write_buffer_, curr_buffer_offset);
    emb_files_[version]->app_count_ += buffer_cur_;
    emb_files_[version]->Flush();
  }

  void CreateFile(size_t version) {
    emb_files_.emplace_back(
        new EmbFile(path_, version, BUFFER_SIZE));
  }

  Status FlushAndUpdate(char* value_buffer, K* id_buffer,
                        EmbPosition** pos_buffer, int64& n_ids,
                        std::vector<int64>& invalid_files) {
    {
      mutex_lock l(mu_);
      compaction_version_ = ++current_version_;
      CreateFile(compaction_version_);
    }

    emb_files_[compaction_version_]->Write(value_buffer, n_ids * val_len_);
    emb_files_[compaction_version_]->app_count_ += n_ids;
    emb_files_[compaction_version_]->Flush();

    for (int64 i = 0; i < n_ids; i++) {
      auto iter = hash_map_.insert_lockless(std::move(
        std::pair<K, EmbPosition*>(id_buffer[i], nullptr)));
      if ((*(iter.first)).first == EMPTY_KEY) {
        return errors::NotFound("Unable to find Key: ",
            id_buffer[i], " in SSDHashKV.");
      } else {
        size_t offset = i * val_len_;
        EmbPosition* ep = new EmbPosition(offset, compaction_version_,
            offset, true);
        bool flag = __sync_bool_compare_and_swap(
            &((*(iter.first)).second), pos_buffer[i], ep);
        if (!flag) {
           __sync_fetch_and_add(
               &emb_files_[compaction_version_]->app_invalid_count, 1);
          if (emb_files_[compaction_version_]->app_count_ >=
                emb_files_[compaction_version_]->app_invalid_count &&
            emb_files_[compaction_version_]->app_count_ / 3 <
                emb_files_[compaction_version_]->app_invalid_count) {
            evict_file_set_.insert_lockless(compaction_version_);
          }
          delete ep;
        } else {
          pos_out_of_date_compact_.emplace_back(pos_buffer[i]);
        }
      }
    }

    for (int i = 0; i < invalid_files.size(); i++) {
      evict_file_set_.erase_lockless(invalid_files[i]);
    }
    invalid_files.clear();
    n_ids = 0;
    return Status::OK();
  }

  void CheckBuffer() {
    size_t curr_buffer_offset = buffer_cur_ * val_len_;
    if (curr_buffer_offset + val_len_ > BUFFER_SIZE) {
      WriteFile(current_version_, curr_buffer_offset);
      if (emb_files_[current_version_]->app_count_ >= max_app_count_) {
        ++current_version_;
        current_offset_ = 0;
        CreateFile(current_version_);
      }
      TF_CHECK_OK(UpdateFlushStatus());
      buffer_cur_ = 0;
    }
  }

  void CheckBufferAsync() {
    size_t curr_buffer_offset = buffer_cur_ * val_len_;
    if (curr_buffer_offset + val_len_ > BUFFER_SIZE) {
      WriteFile(evict_version_, curr_buffer_offset);
      TF_CHECK_OK(UpdateFlushStatus());
      mutex_lock l(mu_);
      evict_version_ = ++current_version_;
      current_offset_ = 0;
      CreateFile(evict_version_);
      buffer_cur_ = 0;
    }
  }

  void AppendToWriteBuffer(size_t curr_buffer_offset, K key,
                            const ValuePtr<V>* value_ptr) {
    current_offset_ += val_len_;
    memcpy(write_buffer_ + curr_buffer_offset,
        (char*)value_ptr->GetPtr(), val_len_);
    key_buffer_[buffer_cur_] = key;
    ++buffer_cur_;
  }

  void AppendToPositionRecordQueue(EmbPosition* old_posi) {
    //A parameter that can be adjusted in the future
    if (pos_out_of_date_.size() > CAP_INVALID_POS) {
      EmbPosition* posi = pos_out_of_date_.front();
      delete posi;
      pos_out_of_date_.pop_front();
    }
    pos_out_of_date_.emplace_back(old_posi);
  }

  bool UpdatePosition(EmbPosition** pos, EmbPosition* old_posi,
      EmbPosition* new_posi) {
    bool flag = __sync_bool_compare_and_swap(pos, old_posi, new_posi);
    if (flag) {
      AppendToPositionRecordQueue(old_posi);
    }
    return flag;
  }

  void SaveKV(K key, const ValuePtr<V>* value_ptr,
      bool is_compaction = false) {
    size_t curr_buffer_offset = buffer_cur_ * val_len_;
    EmbPosition* ep = new EmbPosition(current_offset_, current_version_,
                                      curr_buffer_offset, false);
    AppendToWriteBuffer(curr_buffer_offset, key, value_ptr);

    auto iter = hash_map_.insert_lockless(std::move(
        std::pair<K, EmbPosition*>(key, const_cast<EmbPosition*>(ep))));

    if ((*(iter.first)).second != ep) {
      EmbPosition* old_posi = (*(iter.first)).second;
      int64 version = old_posi->version_;
      if (!is_compaction) {
        emb_files_[version]->app_invalid_count++;
        //A parameter that can be adjusted in the future
        if (version != current_version_ &&
            (emb_files_[version]->app_count_ >=
              emb_files_[version]->app_invalid_count) &&
            (emb_files_[version]->app_count_ / 3 <
              emb_files_[version]->app_invalid_count))
          evict_file_set_.insert_lockless(version);
      }
      UpdatePosition(&((*(iter.first)).second), old_posi, ep);
    }
  }

  void SaveKVAsync(K key, const ValuePtr<V>* value_ptr,
      bool is_compaction = false) {
    size_t curr_buffer_offset = buffer_cur_ * val_len_;
    EmbPosition* ep = new EmbPosition(current_offset_, evict_version_,
                                      curr_buffer_offset, false);

    AppendToWriteBuffer(curr_buffer_offset, key, value_ptr);
    auto iter = hash_map_.insert_lockless(std::move(
        std::pair<K, EmbPosition*>(key, const_cast<EmbPosition*>(ep))));

    if ((*(iter.first)).second != ep) {
      bool flag = false;
      EmbPosition* old_posi = nullptr;
      do {
        old_posi = (*(iter.first)).second;
        flag = UpdatePosition(&((*(iter.first)).second), old_posi, ep);
      } while (!flag);

      if (!is_compaction) {
        int version = old_posi->version_;
        __sync_fetch_and_add(&emb_files_[version]->app_invalid_count, 1);
        //A parameter that can be adjusted in the future
        if (version != evict_version_ &&
            emb_files_[version]->app_count_ >=
            emb_files_[version]->app_invalid_count &&
            emb_files_[version]->app_count_ / 3 <
            emb_files_[version]->app_invalid_count) {
          evict_file_set_.insert_lockless(version);
        }
      }
    }
  }

  void DeleteInvalidFiles() {
    for (auto it : evict_file_map_) {
      emb_files_[it.first]->DeleteFile();
    }
    evict_file_map_.clear();
  }

  void DeleteInvalidRecord() {
    for (auto it: pos_out_of_date_compact_) {
      delete it;
    }
    pos_out_of_date_compact_.clear();
  }

  void LookupValidItems() {
    for (auto it : hash_map_) {
      EmbPosition* posi = it.second;
      auto iter = evict_file_map_.find(posi->version_);
      if (iter != evict_file_map_.end()) {
        (*iter).second.emplace_back(it);
      }
    }
  }

  void InitializeEvictMap() {
    for (auto it : evict_file_set_) {
      std::vector<std::pair<K, EmbPosition*>> tmp;
      evict_file_map_[it] = tmp;
      evict_file_set_.erase_lockless(it);
    }
    LookupValidItems();
  }

  void InitializeEvictMapWithoutErase() {
    for (auto it : evict_file_set_) {
      std::vector<std::pair<K, EmbPosition*>> tmp;
      evict_file_map_[it] = tmp;
    }
    LookupValidItems();
  }

  void MoveToNewFile() {
    ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
    for (auto it : evict_file_map_) {
      EmbFile* file = emb_files_[it.first];
      total_app_count_ -= file->app_invalid_count;
      file->Map();
      for (auto it_vec : it.second) {
        EmbPosition* posi = it_vec.second;
        file->ReadWithoutMap((char*)(val->GetPtr()), val_len_,
            posi->offset_);
        CheckBuffer();
        SaveKV(it_vec.first, val, true);
      }
      file->Unmap();
    }
    delete val;
  }

  void MoveToNewFileAsync() {
    char* compact_buffer = new char[BUFFER_SIZE];
    int64 n_ids = 0;
    std::vector<int64> invalid_files;
    unsigned int max_key_count = 1 + int(BUFFER_SIZE / val_len_);
    K id_buffer[max_key_count] = {0};
    EmbPosition* pos_buffer[max_key_count] = {0};
    for (auto it : evict_file_map_) {
      EmbFile* file = emb_files_[it.first];
      __sync_fetch_and_sub(&total_app_count_, file->app_invalid_count);
      file->Map();
      for (auto it_vec : it.second) {
        EmbPosition* posi = it_vec.second;
        id_buffer[n_ids] = it_vec.first;
        pos_buffer[n_ids] = posi;
        file->ReadWithoutMap(compact_buffer + val_len_ * n_ids, val_len_,
            posi->offset_);
        n_ids++;
        if (n_ids == max_app_count_) {
          Status st = FlushAndUpdate(compact_buffer, id_buffer,
              pos_buffer, n_ids, invalid_files);
          if(!st.ok()) {
            LOG(WARNING)<<"FLUSH ERROR: "<<st.ToString();
          }
        }
      }
      file->Unmap();
      invalid_files.emplace_back(it.first);
    }
    Status st = FlushAndUpdate(compact_buffer, id_buffer,
        pos_buffer, n_ids, invalid_files);
    if(!st.ok()) {
      LOG(WARNING)<<"FLUSH ERROR: "<<st.ToString();
    }
    delete[] compact_buffer;
  }

  void Compaction() {
    int64 hash_size = hash_map_.size_lockless();
    //These parameter that can be adjusted in the future
    if (hash_size * 3 / 2 < total_app_count_ ||
        total_app_count_ - hash_size > CAP_INVALID_ID) {
      // delete the evict_files
      DeleteInvalidFiles();
      // Initialize evict_file_map
      InitializeEvictMap();
      // read embeddings and write to new file
      MoveToNewFile();
    }
  }

  void CompactionAsync() {
    int64 hash_size = hash_map_.size_lockless();
    //These parameter that can be adjusted in the future
    if (hash_size * 3 / 2 < total_app_count_ ||
        total_app_count_ - hash_size > CAP_INVALID_ID) {
      DeleteInvalidRecord();
      // delete the evict_files
      DeleteInvalidFiles();
      // Initialize evict_file_map
      InitializeEvictMapWithoutErase();
      // read embeddings and write to new file
      MoveToNewFileAsync();
    }
  }

  void CompactionThread() {
    if (val_len_ == -1) {
      while (!done_) {
      }
    }
    while (!shutdown_) {
      CompactionAsync();
    }
  }

  std::string DebugString() const {
    return strings::StrCat("map info size:", Size(),
                           ", map info bucket_count:",
                           hash_map_.load_factor(),
                           ",map info load_factor:",
                           hash_map_.load_factor(),
                           ", map info max_load_factor:",
                           hash_map_.max_load_factor(),
                           ", map info min_load_factor: ",
                           hash_map_.min_load_factor(),
                           ", evict_version: ", evict_version_,
                           ", compaction_version: ", compaction_version_);
  }

 private:
  size_t val_len_;
  size_t current_version_;
  size_t evict_version_;
  size_t compaction_version_;
  size_t current_offset_;
  size_t buffer_cur_;
  size_t total_app_count_;
  size_t max_app_count_;

  char* write_buffer_;
  K* key_buffer_;
  bool is_async_compaction_;
  Allocator* alloc_;

  int total_dims_;
  std::string path_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;

  typedef google::dense_hash_map_lockless<K, EmbPosition*> LockLessHashMap;
  LockLessHashMap hash_map_;
  mutex mu_;

  static constexpr int EMPTY_KEY = -1;
  static constexpr int CAP_INVALID_POS = 200000;
  static constexpr int CAP_INVALID_ID = 10000000;
  static constexpr size_t BUFFER_SIZE = 1 << 27;

  std::vector<EmbFile*> emb_files_;
  std::deque<EmbPosition*> pos_out_of_date_;
  std::deque<EmbPosition*> pos_out_of_date_compact_;
  typedef google::dense_hash_set_lockless<K> LocklessHashSet;
  LocklessHashSet evict_file_set_;
  std::map<int64, std::vector<std::pair<K, EmbPosition*>>> evict_file_map_;

  Thread* compaction_thread_;
  condition_variable shutdown_cv_;
  volatile bool shutdown_ = false;
  volatile bool done_ = false;
  // std::atomic_flag flag_ = ATOMIC_FLAG_INIT; unused

  std::function<void()> compaction_fn_;
  std::function<void()> check_buffer_fn_;
  std::function<void(K, const ValuePtr<V>*, bool)> save_kv_fn_;
};

}  // namespace embedding
}  // namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASHKV_H_

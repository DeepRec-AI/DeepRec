#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_

#include <cmath>
#include "tensorflow/core/framework/embedding/config.pb.h"

namespace tensorflow {
struct EmbeddingConfig {
  int64 emb_index;
  int64 primary_emb_index;
  int64 block_num;
  int64 slot_num;
  std::string name;
  int64 steps_to_live;
  int64 filter_freq;
  int64 max_freq;
  float l2_weight_threshold;
  LayoutType layout_type;
  int64 kHashFunc;
  int64 num_counter;
  DataType counter_type;
  embedding::StorageType storage_type;
  std::string storage_path;
  int64 storage_size;
  int64 default_value_dim;
  int normal_fix_flag;

  EmbeddingConfig(int64 emb_index = 0, int64 primary_emb_index = 0,
                  int64 block_num = 1, int slot_num = 0,
                  const std::string& name = "", int64 steps_to_live = 0,
                  int64 filter_freq = 0, int64 max_freq = 999999,
                  float l2_weight_threshold = -1.0, const std::string& layout = "normal",
                  int64 max_element_size = 0, float false_positive_probability = -1.0,
                  DataType counter_type = DT_UINT64, embedding::StorageType storage_type = embedding::DRAM,
                  const std::string& storage_path = "", int64 storage_size = 0,
                  int64 default_value_dim = 4096):
      emb_index(emb_index),
      primary_emb_index(primary_emb_index),
      block_num(block_num),
      slot_num(slot_num),
      name(name),
      steps_to_live(steps_to_live),
      filter_freq(filter_freq),
      max_freq(max_freq),
      l2_weight_threshold(l2_weight_threshold),
      counter_type(counter_type),
      storage_type(storage_type),
      storage_path(storage_path),
      storage_size(storage_size),
      default_value_dim(default_value_dim),
      normal_fix_flag(0) {
    if ("normal" == layout) {
      layout_type = LayoutType::NORMAL;
    } else if ("light" == layout) {
      layout_type = LayoutType::LIGHT;
    } else if ("normal_fix" == layout){
      layout_type = LayoutType::NORMAL_FIX;
    } else {
      LOG(WARNING) << "Unknown layout: " << layout << ", use LayoutType::NORMAL by default.";
      layout_type = LayoutType::NORMAL;
    }
    if (max_element_size != 0 && false_positive_probability != -1.0){
      kHashFunc = calc_num_hash_func(false_positive_probability);
      num_counter = calc_num_counter(max_element_size, false_positive_probability); 
    } else {
      kHashFunc = 0;
      num_counter = 0;
    }
    if (layout_type == LayoutType::NORMAL_FIX) {
      normal_fix_flag = 1;
    }
  }
  
  int64 calc_num_counter(int64 max_element_size, float false_positive_probability) {
    float loghpp = fabs(log(false_positive_probability));
    float factor = log(2) * log(2);
    return ceil(loghpp / factor * max_element_size);
  }

  int64 calc_num_hash_func(float false_positive_probability) {
    float loghpp = fabs(log(false_positive_probability)/log(2));
    return ceil(loghpp);
  }
  bool is_primary() const {
    return emb_index == primary_emb_index;
  }

  int64 total_num(int total_dims) {
    return block_num * (1 + (1 - normal_fix_flag) * (slot_num + 1)) * (1 + normal_fix_flag * (total_dims - 1));
  }

  int64 get_filter_freq() {
    return filter_freq;
  }

  LayoutType get_layout_type() {
    return layout_type;
  }

  embedding::StorageType get_storage_type() {
    return storage_type;
  }

  std::string get_storage_path() {
    return storage_path;
  }

  int64 get_storage_size() {
    return storage_size;
  }

  std::string DebugString() const {
    return strings::StrCat("opname: ", name,
                           " emb_index: ", emb_index,
                           " primary_emb_index: ", primary_emb_index,
                           " block_num: ", block_num,
                           " slot_num: ", slot_num,
                           " layout_type: ", static_cast<int>(layout_type),
                           " steps_to_live: ", steps_to_live,
                           " filter_freq: ", filter_freq,
                           " max_freq: ", max_freq,
                           " l2_weight_threshold: ", l2_weight_threshold,
                           " storage_type: ", storage_type,
                           " storage_path: ", storage_path,
                           " storage_size: ", storage_size);
  }
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_


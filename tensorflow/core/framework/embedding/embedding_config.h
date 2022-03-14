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
  int64 kHashFunc;
  int64 num_counter;
  DataType counter_type;
  int64 default_value_dim;
  int normal_fix_flag;

  EmbeddingConfig(int64 emb_index = 0, int64 primary_emb_index = 0,
                  int64 block_num = 1, int slot_num = 0,
                  const std::string& name = "", int64 steps_to_live = 0,
                  int64 filter_freq = 0, int64 max_freq = 999999,
                  float l2_weight_threshold = -1.0, const std::string& layout = "normal",
                  int64 max_element_size = 0, float false_positive_probability = -1.0,
                  DataType counter_type = DT_UINT64,
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
      default_value_dim(default_value_dim),
      normal_fix_flag(0) {
    if (max_element_size != 0 && false_positive_probability != -1.0){
      kHashFunc = calc_num_hash_func(false_positive_probability);
      num_counter = calc_num_counter(max_element_size, false_positive_probability);
    } else {
      kHashFunc = 0;
      num_counter = 0;
    }
    if (layout == "normal_contiguous") {
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

  int64 total_num(int alloc_len) {
    return block_num * (1 + (1 - normal_fix_flag) * slot_num) * (1 + normal_fix_flag * (alloc_len * (slot_num + 1) - 1));
  }

  int64 get_filter_freq() {
    return filter_freq;
  }

  std::string DebugString() const {
    return strings::StrCat("opname: ", name,
                           " emb_index: ", emb_index,
                           " primary_emb_index: ", primary_emb_index,
                           " block_num: ", block_num,
                           " slot_num: ", slot_num,
                           " steps_to_live: ", steps_to_live,
                           " filter_freq: ", filter_freq,
                           " max_freq: ", max_freq,
                           " l2_weight_threshold: ", l2_weight_threshold);
  }
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_CONFIG_H_


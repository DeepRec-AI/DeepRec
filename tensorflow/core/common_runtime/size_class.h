#ifndef TENSORFLOW_COMMON_RUNTIME_SIZE_CLASS_H_
#define TENSORFLOW_COMMON_RUNTIME_SIZE_CLASS_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
constexpr int kClassNum = 67;
constexpr int kMaxClassSize = 32 * 1024; // 32 KB
// This size class is token from tcmalloc
const int kSizeClass[kClassNum] = {
         0,
         8,
        16,
        32,
        48,
        64,
        80,
        96,
       112,
       128,
       144,
       160,
       176,
       192,
       208,
       224,
       240,
       256,
       272,
       288,
       304,
       320,
       336,
       352,
       368,
       384,
       400,
       416,
       448,
       480,
       512,
       576,
       640,
       704,
       768,
       896,
      1024,
      1152,
      1280,
      1408,
      1536,
      1792,
      2048,
      2304,
      2688,
      2816,
      3200,
      3456,
      3584,
      4096,
      4736,
      5376,
      6144,
      6528,
      6784,
      7168,
      8192,
      9472,
     10240,
     12288,
     13568,
     14336,
     16384,
     20480,
     24576,
     28672,
     32768 };

class SizeMap {
public:
  
  //-------------------------------------------------------------------
  // Mapping from size to size_class and vice versa
  //-------------------------------------------------------------------

  // Sizes <= 1024 have an alignment >= 8.  So for such sizes we have an
  // array indexed by ceil(size/8).  Sizes > 1024 have an alignment >= 128.
  // So for these larger sizes we have an array indexed by ceil(size/128).
  //
  // We flatten both logical arrays into one physical array and use
  // arithmetic to compute an appropriate index.  The constants used by
  // ClassIndex() were selected to make the flattening work.
  //
  // Examples:
  //   Size       Expression                      Index
  //   -------------------------------------------------------
  //   0          (0 + 7) / 8                     0
  //   1          (1 + 7) / 8                     1
  //   ...
  //   1024       (1024 + 7) / 8                  128
  //   1025       (1025 + 127 + (120<<7)) / 128   129
  //   ...
  //   32768      (32768 + 127 + (120<<7)) / 128  376
  static constexpr int kMaxSmallSize = 1024;
  static constexpr size_t kClassArraySize =
      ((kMaxClassSize + 127 + (120 << 7)) >> 7) + 1;

  SizeMap() {
    int next_size = 0;
    for (int c = 1; c < kClassNum; c++) {
      const int max_size_in_class = kSizeClass[c];

      for (int s = next_size; s <= max_size_in_class; s += 8) {
        class_array_[ClassIndex(s)] = c;
      }
      next_size = max_size_in_class + 8;
      if (next_size > kMaxClassSize) {
        break;
      }
    }
  }


  inline size_t ClassIndex(size_t size) const {
    if (size <= kMaxSmallSize) {
      return (size + 7) >> 3;
    } else if (size <= kMaxClassSize) {
      return (size + 127 + (120 << 7)) >> 7;
    }
    LOG(ERROR) << "size " << size << " out of range";
    return 0;
  }

  inline size_t GetClass(size_t size) const {
    return class_array_[ClassIndex(size)];
  }

 private:
  int class_array_[kClassArraySize];
};

} // namespace tensorflow

#endif // TENSORFLOW_COMMON_RUNTIME_SIZE_CLASS_H_

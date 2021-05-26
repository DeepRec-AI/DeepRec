#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_STAT_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_STAT_H_

#include <cstddef>
#include <sys/time.h>


namespace tensorflow {
namespace {
const static size_t COUNTER_INTERVAL = 1000;
} // namespace

class SeastarStat {
public:
  explicit SeastarStat();
  virtual ~SeastarStat() {}

  void TensorResponse(size_t tensor_size);
  void Request();

private:
  void Setup();

private:
  timeval _start;
  size_t _counter;
  double _total_tensor_size;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_STAT_H_

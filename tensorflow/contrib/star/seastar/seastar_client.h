#ifndef TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_CLIENT_H_
#define TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_CLIENT_H_

#include <iostream>
#include <string>

#include "tensorflow/contrib/star/seastar/seastar_header.h"

namespace seastar {
class channel;
}

namespace tensorflow {
class SeastarTagFactory;
class SeastarClientTag;

class SeastarClient {
public:
  // should named start & stop which used by seastar template class distributed<>
  void start(seastar::ipv4_addr server_addr,
             std::string s,
             seastar::channel* chan,
             SeastarTagFactory* tag_factory);

  seastar::future<> stop();

private:
  struct Connection {
    seastar::connected_socket _fd;
    seastar::input_stream<char> _read_buf;
    seastar::channel* _channel;
    SeastarTagFactory* _tag_factory;
    seastar::socket_address _addr;
    Connection(seastar::connected_socket&& fd,
               seastar::channel* chan,
               SeastarTagFactory* tag_factory,
               seastar::socket_address addr);
    seastar::future<> Read();
    seastar::future<> ReapeatReadTensors(
        SeastarClientTag* tag, int* count,
        int* cur_idx, bool* error);
  };
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SEASTAR_SEASTAR_CLIENT_H_

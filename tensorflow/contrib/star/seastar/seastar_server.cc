#include <errno.h>
#include <exception>

#include "boost/asio/ip/address_v4.hpp"
#include "seastar/core/channel.hh"
#include "seastar/core/reactor.hh"
#include "tensorflow/contrib/star/seastar/seastar_server.h"
#include "tensorflow/contrib/star/seastar/seastar_server_tag.h"
#include "tensorflow/contrib/star/seastar/seastar_tag_factory.h"
#include "tensorflow/contrib/star/star_message.h"
#include "tensorflow/core/platform/logging.h"


namespace tensorflow {

void SeastarServer::start(uint16_t port, SeastarTagFactory* tag_factory) {
  seastar::listen_options lo;
  lo.reuse_address = true;
  try {
    _listener = seastar::engine().listen(seastar::make_ipv4_address(port), lo);
  } catch (std::exception &e) {
    LOG(ERROR) << "Error code: " << errno << ", " << e.what();
  }

  seastar::keep_doing([this, tag_factory] {
    return _listener->accept().then(
        [this, tag_factory]
        (seastar::connected_socket fd, seastar::socket_address addr) mutable {
      auto conn = new Connection(std::move(fd), tag_factory, addr);

      VLOG(1) << "Seastar engine accept a connection, remote add: " << addr
              << ", cpu id:" << seastar::engine().cpu_id();
      seastar::do_until([conn] {return conn->_read_buf.eof(); }, [conn] {
        return conn->Read();
      }).then_wrapped([this, conn] (auto&& f) {
        try {
          f.get();
          LOG(WARNING) << "Connection closed by remote peer, remote addr: "
                       << conn->_addr;
        } catch (std::exception& ex) {
          LOG(WARNING) << ex.what()
                       << " errno: " << errno
                       << ", remote addr: " << conn->_addr;
        }
      });
    });
  }).or_terminate();
}
  
seastar::future<> SeastarServer::stop() {
  return seastar::make_ready_future<>();
}

SeastarServer::Connection::Connection(seastar::connected_socket&& fd,
    SeastarTagFactory* tag_factory, seastar::socket_address addr)
  : _tag_factory(tag_factory),
    _addr(addr) {
  seastar::ipv4_addr ip_addr(addr);
  boost::asio::ip::address_v4 addr_v4(ip_addr.ip);
  string addr_str = addr_v4.to_string() + ":" + std::to_string(ip_addr.port);
  _channel = new seastar::channel(addr_str);
  _fd = std::move(fd);
  _fd.set_nodelay(true);
  _read_buf = _fd.input();
  _channel->init(seastar::engine().get_packet_queue(), std::move(_fd.output()));
}

SeastarServer::Connection::~Connection() {
  delete _channel;
}

seastar::future<> SeastarServer::Connection::Read() {
  return _read_buf.read_exactly(StarServerTag::kHeaderSize)
    .then([this] (auto&& header) {
        CHECK_CONNECTION_CLOSE(header.size());

        if (header.size() != StarServerTag::kHeaderSize)
          return seastar::make_ready_future();

        auto tag = _tag_factory->CreateSeastarServerTag(header, _channel);
        auto req_body_size = tag->GetRequestBodySize();
        if (req_body_size == 0) {
          tag->RecvReqDone(tensorflow::Status());
          return seastar::make_ready_future();
        }
          
        auto req_body_buffer = tag->GetRequestBodyBuffer();
        return _read_buf.read_exactly(req_body_size)
          .then([this, tag, req_body_size, req_body_buffer] (auto&& body) {
              CHECK_CONNECTION_CLOSE(body.size());
              if (req_body_size != body.size()) {
                LOG(WARNING) << "warning expected body size is:"
                             << req_body_size << ", actual body size:" << body.size();
                tag->RecvReqDone(tensorflow::Status(error::UNKNOWN, 
                                                    "Seastar Server: read invalid msgbuf"));
                return seastar::make_ready_future<>();
              }

              if (tag->IsStarRunGraph()) {
                InitStarServerTag(tag);
                tag->ParseMetaData(body.get(), body.size());

                int* recv_count = new int(tag->GetReqTensorCount());
                int* idx = new int(0);
                bool *error = new bool(false);
                return this->ReapeatReadTensors(tag, recv_count, idx, error);

              } else {
                memcpy(req_body_buffer, body.get(), body.size());
                tag->RecvReqDone(tensorflow::Status());
                return seastar::make_ready_future();
              }
            });
      });
}

seastar::future<> SeastarServer::Connection::ReapeatReadTensors(
    SeastarServerTag* tag, int* count, int* idx, bool* error) {
  return seastar::do_until(
      [this, tag, count, idx, error] {
        if (*error || *idx == *count) {
          delete count;
          delete idx;
          if (!(*error)) {
            tag->ParseTensor();
            tag->RecvReqDone(tensorflow::Status());
          }
          delete error;
          return true;
        } else {
          return false;
        }
      },
      [this, tag, idx, error] {
        return _read_buf.read_exactly(StarMessage::kMessageTotalBytes)
          .then([this, tag, idx, error] (auto&& tensor_msg) {
            CHECK_CONNECTION_CLOSE(tensor_msg.size());
            tag->ParseMessage(*idx, tensor_msg.get(), tensor_msg.size());
            auto tensor_size = tag->GetRequestTensorSize(*idx);
            auto tensor_buffer = tag->GetRequestTensorBuffer(*idx);
            ++(*idx);

            // This will not quit seastar::do_until,
            // it will recv other tensors.    
            if (tensor_size == 0) {
               return seastar::make_ready_future();
            }

            if (tensor_size >= _8KB) {
              return _read_buf.read_exactly(tensor_buffer, tensor_size)
                .then([this, tag, error, tensor_size, tensor_buffer] (auto read_size) {
                  CHECK_CONNECTION_CLOSE(read_size);
                  if (read_size != tensor_size) {
                    LOG(WARNING) << "warning expected read size is:" << tensor_size
                                 << ", actual read tensor size:" << read_size;
                    tag->RecvReqDone(
                        tensorflow::Status(error::UNKNOWN,
                                           "Seastar Server: read invalid tensorbuf"));
                    *error = true;
                    return seastar::make_ready_future();
                  }
                  // No need copy here
                  return seastar::make_ready_future();
                });
            } else {
              return _read_buf.read_exactly(tensor_size)
                .then([this, tag, error, tensor_size, tensor_buffer] (auto&& tensor) {
                  CHECK_CONNECTION_CLOSE(tensor.size());
                  if (tensor.size() != tensor_size) {
                    LOG(WARNING) << "warning expected read size is:" << tensor_size
                                 << ", actual read tensor size:" << tensor.size();
                    tag->RecvReqDone(
                        tensorflow::Status(error::UNKNOWN,
                                           "Seastar Server: read invalid tensorbuf"));
                    *error = true;
                    return seastar::make_ready_future();
                  }
                  memcpy(tensor_buffer, tensor.get(), tensor.size());
                  return seastar::make_ready_future();
                });
            }
        });
      });
}

} // namespace tensorflow

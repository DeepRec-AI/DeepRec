#include "tensorflow/contrib/star/seastar/seastar_client.h"
#include "tensorflow/contrib/star/seastar/seastar_header.h"
#include "tensorflow/contrib/star/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/star/seastar/seastar_tag_factory.h"
#include "tensorflow/contrib/star/star_message.h"
#include "tensorflow/core/platform/logging.h"

using namespace std::chrono_literals;

namespace tensorflow {

SeastarClient::Connection::Connection(seastar::connected_socket&& fd,
                                      seastar::channel* chan,
                                      SeastarTagFactory* tag_factory,
                                      seastar::socket_address addr)
  : _channel(chan), _tag_factory(tag_factory), _addr(addr) {
  _fd = std::move(fd);
  _fd.set_nodelay(true);
  _read_buf = _fd.input();
  _channel->init(seastar::engine().get_packet_queue(), std::move(_fd.output()));
}

seastar::future<> SeastarClient::Connection::Read() {
  return _read_buf.read_exactly(StarClientTag::kHeaderSize).then([this] (auto&& header) {
      CHECK_CONNECTION_CLOSE(header.size());

      auto tag = _tag_factory->CreateSeastarClientTag(header);
      if (tag->status_ != 0) {
        return _read_buf.read_exactly(tag->err_msg_len_).then([this, tag] (auto&& err_msg) {
          CHECK_CONNECTION_CLOSE(err_msg.size());
          std::string msg = std::string(err_msg.get(), tag->err_msg_len_);
          if (msg.empty()) {
            msg = "Empty error msg.";
          }
          tensorflow::Status s(static_cast<tensorflow::error::Code>(tag->status_),  msg);
          tag->ScheduleProcess([tag, s] {
              tag->HandleResponse(s);
            });

          return seastar::make_ready_future();
        });
      }

      if (tag->IsStarRunGraph()) {
        // Handle zero copy run graph response.
        auto resp_meta_size = tag->GetResponseBodySize();
        if (resp_meta_size == 0) {
          tag->ScheduleProcess([tag] {
              tag->HandleResponse(tensorflow::Status());
            });
          return seastar::make_ready_future();
        }

        auto resp_meta_buffer = tag->GetResponseBodyBuffer();
        return _read_buf.read_exactly(resp_meta_size)
          .then([this, tag, resp_meta_size, resp_meta_buffer](auto&& meta) {
            CHECK_CONNECTION_CLOSE(meta.size());
            if (meta.size() != resp_meta_size) {
              LOG(ERROR) << "warning expected read size is:" << resp_meta_size
                         << ", meta data size:" << meta.size();
              tag->ScheduleProcess([tag] {
                  tag->HandleResponse(tensorflow::Status(error::UNKNOWN,
                                                         "Seastar Client read invalid meta data."));
                });

              return seastar::make_ready_future();
            }

            tag->ParseStarRunGraphMeta(meta.get(), meta.size());

            int* recv_count = new int(tag->resp_tensor_count_);
            int* idx = new int(0);
            bool *error = new bool(false);
            return this->ReapeatReadTensors(tag, recv_count, idx, error);
        });

      } else if (tag->IsRecvTensor()) {
        // Handle recv_tensor/fuse_recv_tensor response
        int *recv_count = new int(tag->resp_tensor_count_);
        int *idx = new int(0);
        bool *error = new bool(false);
        return this->ReapeatReadTensors(tag, recv_count, idx, error);

      } else {
        // Handle other response, which has a pb payload.
        auto resp_body_size = tag->GetResponseBodySize();
        if (resp_body_size == 0) {
          tag->ScheduleProcess([tag] {
              tag->HandleResponse(tensorflow::Status());
            });
          return seastar::make_ready_future();
        }

        auto resp_body_buffer = tag->GetResponseBodyBuffer();
        return _read_buf.read_exactly(resp_body_size)
          .then([this, tag, resp_body_size, resp_body_buffer](auto&& body) {
              CHECK_CONNECTION_CLOSE(body.size());
              if (body.size() != resp_body_size) {
                LOG(ERROR) << "warning expected read size is:" << resp_body_size
                           << ", body size:" << body.size();
                tag->ScheduleProcess([tag] {
                    tag->HandleResponse(tensorflow::Status(error::UNKNOWN,
                                                           "Seastar Client read invalid resp."));
                  });
                return seastar::make_ready_future();
              }
              memcpy(resp_body_buffer, body.get(), body.size());
              tag->ScheduleProcess([tag] {
                  tag->HandleResponse(tensorflow::Status());
                });
              return seastar::make_ready_future();
            });
      }
    });
}

seastar::future<> SeastarClient::Connection::ReapeatReadTensors(SeastarClientTag* tag,
                                                                int* count,
                                                                int* idx,
                                                                bool* error) {
  return seastar::do_until(
      [this, tag, count, idx, error] {
        if (*error || *idx == *count) {
          delete count;
          delete idx;
          // NOTE(rangeng.llb): If error happens, tag->RecvRespDone has been called.
          if (!(*error)) {
            tag->ScheduleProcess([tag] {
                tag->HandleResponse(tensorflow::Status());
              });
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
            tag->ParseTensorMessage(*idx, tensor_msg.get(), tensor_msg.size());
            auto tensor_size = tag->GetResponseTensorSize(*idx);
            auto tensor_buffer = tag->GetResponseTensorBuffer(*idx);
            ++(*idx);
                    
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
                    tag->ScheduleProcess([tag] {
                        tag->HandleResponse(tensorflow::Status(error::UNKNOWN,
                                                               "Seastar Client: read invalid tensorbuf"));
                    });
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
                    tag->ScheduleProcess([tag] {
                        tag->HandleResponse(tensorflow::Status(error::UNKNOWN,
                                                               "Seastar Client: read invalid tensorbuf"));
                      });
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

void SeastarClient::start(seastar::ipv4_addr server_addr, std::string s, seastar::channel* chan, SeastarTagFactory* tag_factory) {
  seastar::socket_address local = seastar::socket_address(::sockaddr_in{AF_INET, INADDR_ANY, {0}});
  seastar::engine().net().connect(seastar::make_ipv4_address(server_addr), local, seastar::transport::TCP).then(
      [this, chan, tag_factory, s, server_addr] (seastar::connected_socket fd) {
      auto conn = new Connection(std::move(fd), chan, tag_factory, seastar::socket_address(server_addr));

      //LOG(INFO) << "connected...." << s;
      seastar::do_until([conn] {return conn->_read_buf.eof(); }, [conn] {
        return conn->Read();
      }).then_wrapped([this, conn, s, chan] (auto&& f) {
        try {
          f.get();
          LOG(INFO) << "Remote closed the connection: addr = " << s;
        } catch(std::exception& ex) {
          LOG(INFO) << ex.what() << ", errno=" << errno << ", addr = " << s;
        }
        LOG(INFO) << "Set channel broken, connection:" << s;
        chan->set_channel_broken();
      });

      return seastar::make_ready_future();
    }).handle_exception([this, chan, tag_factory, server_addr, s](auto ep) {
      return seastar::sleep(1s).then([this, chan, tag_factory, server_addr, s] {
        //LOG(INFO) << "connected failure...." << s;
        this->start(server_addr, s, chan, tag_factory);
      });
    });
}

seastar::future<> SeastarClient::stop() {
  return seastar::make_ready_future();
}

} // namespace tensorflow

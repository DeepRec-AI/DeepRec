/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <deque>
#include <unordered_map>

#include "rdkafkacpp.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace data {

class KafkaDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* topics_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("topics", &topics_tensor));
    OP_REQUIRES(
        ctx, topics_tensor->dims() <= 1,
        errors::InvalidArgument("`topics` must be a scalar or a vector."));

    std::vector<string> topics;
    topics.reserve(topics_tensor->NumElements());
    for (int i = 0; i < topics_tensor->NumElements(); ++i) {
      topics.push_back(topics_tensor->flat<string>()(i));
    }

    string servers = "";
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<string>(ctx, "servers", &servers));
    string group = "";
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "group", &group));
    bool eof = false;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "eof", &eof));
    int64 timeout = -1;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "timeout", &timeout));
    OP_REQUIRES(ctx, (timeout > 0),
                errors::InvalidArgument(
                    "Timeout value should be large than 0, got ", timeout));

    const Tensor* config_global_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("config_global", &config_global_tensor));
    std::vector<string> config_global;
    config_global.reserve(config_global_tensor->NumElements());
    for (int i = 0; i < config_global_tensor->NumElements(); ++i) {
      config_global.push_back(config_global_tensor->flat<string>()(i));
    }

    const Tensor* config_topic_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("config_topic", &config_topic_tensor));
    std::vector<string> config_topic;
    config_topic.reserve(config_topic_tensor->NumElements());
    for (int i = 0; i < config_topic_tensor->NumElements(); ++i) {
      config_topic.push_back(config_topic_tensor->flat<string>()(i));
    }
    bool message_key = false;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<bool>(ctx, "message_key", &message_key));

    *output = new Dataset(ctx, std::move(topics), servers, group, eof, timeout,
                          std::move(config_global), std::move(config_topic),
                          message_key);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> topics,
            const string& servers, const string& group, const bool eof,
            const int64 timeout, std::vector<string> config_global,
            std::vector<string> config_topic, const bool message_key)
        : DatasetBase(DatasetContext(ctx)),
          topics_(std::move(topics)),
          servers_(servers),
          group_(group),
          eof_(eof),
          timeout_(timeout),
          config_global_(std::move(config_global)),
          config_topic_(std::move(config_topic)),
          message_key_(message_key) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Kafka")}));
    }

    const DataTypeVector& output_dtypes() const override {
      if (message_key_) {
        static DataTypeVector* dtypes =
            new DataTypeVector({DT_STRING, DT_STRING});
        return *dtypes;
      }
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      if (message_key_) {
        static std::vector<PartialTensorShape>* shapes =
            new std::vector<PartialTensorShape>({{}, {}});
        return *shapes;
      }
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override { return "KafkaDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* topics = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(topics_, &topics));
      Node* servers = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(servers_, &servers));
      Node* group = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(group_, &group));
      Node* eof = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(eof_, &eof));
      Node* timeout = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(timeout_, &timeout));
      Node* config_global = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(config_global_, &config_global));
      Node* config_topic = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(config_topic_, &config_topic));
      Node* message_key = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(message_key_, &message_key));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this,
                        {topics, servers, group, eof, timeout, config_global,
                         config_topic, message_key},
                        output));
      return Status::OK();
    }

   private:
    class KafkaEventCb : public RdKafka::EventCb {
     public:
      KafkaEventCb(bool& run) : run_(run) {}

      void event_cb(RdKafka::Event& event) {
        switch (event.type()) {
          case RdKafka::Event::EVENT_ERROR:
            LOG(ERROR) << "EVENT_ERROR: "
                       << "(" << RdKafka::err2str(event.err())
                       << "): " << event.str();
            { run_ = !event.fatal(); }
            break;

          case RdKafka::Event::EVENT_STATS:
            LOG(ERROR) << "EVENT_STATS: " << event.str();
            break;

          case RdKafka::Event::EVENT_LOG:
            LOG(ERROR) << "EVENT_LOG: " << event.severity() << "-"
                       << event.fac().c_str() << "-" << event.str().c_str();
            break;

          case RdKafka::Event::EVENT_THROTTLE:
            LOG(ERROR) << "EVENT_THROTTLE: " << event.throttle_time()
                       << "ms by " << event.broker_name() << " id "
                       << (int)event.broker_id();
            break;

          default:
            LOG(ERROR) << "EVENT: " << event.type() << " ("
                       << RdKafka::err2str(event.err()) << "): " << event.str();
            break;
        }
      }

     private:
      bool& run_;
    };

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
      }

      virtual ~Iterator() { ResetStreamsLocked(); }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        if (!init_) {
          TF_RETURN_IF_ERROR(SetupStreamsLocked(nullptr));
          init_ = true;
        }

        while (run_) {
          int64 ts_min = -1;
          int index = -1, i = 0;
          for (auto& iter : consumer_infos_) {
            if (iter.limit_ >= 0 &&
                (iter.topic_partition_->offset() >= iter.limit_ ||
                 iter.offset_ >= iter.limit_)) {
              // EOF current topic
              iter.eof_ = true;
              continue;
            }
            auto& message = iter.message_;
            if (message == nullptr ||
                //message->err() == RdKafka::ERR__PARTITION_EOF ||
                message->err() == RdKafka::ERR__TIMED_OUT) {
              message.reset(
                  iter.consumer_->consume(dataset()->timeout_));
            }
            if (message->err() == RdKafka::ERR_NO_ERROR) {
                RdKafka::MessageTimestamp ts = message->timestamp();
                if (ts_min == -1) {
                  ts_min = ts.timestamp;
                  index = i;
                } else if (ts.timestamp < ts_min) {
                  ts_min = ts.timestamp;
                  index = i;
                }
            } else if (message->err() == RdKafka::ERR__PARTITION_EOF) {
              LOG(INFO) << "Partition reach EOF: " << iter.topic_partition_->topic()
                        << ", partition: " << iter.topic_partition_->partition()
                        << ", current offset: " << iter.offset_;
              message.reset(nullptr);
              if (dataset()->eof_) {
                iter.eof_ = true;
              }
            } else if (message->err() == RdKafka::ERR__TRANSPORT) {
              // Not return error here because consumer will try re-connect.
              LOG(ERROR) << "Broker transport failure: " << message->errstr();
            } else if (message->err() != RdKafka::ERR__TIMED_OUT) {
              LOG(ERROR) << "Failed to consume: " << message->errstr();
              return errors::Internal("Failed to consume: ",
                                      message->errstr());
            }
            ++i;
          }
          if (index == -1) {
            bool should_stop = true;
            for (auto& iter : consumer_infos_) {
              should_stop &= iter.eof_;
            }
            if (should_stop) {
              *end_of_sequence = true;
              return Status::OK();
            }
          } else {
            auto& message = consumer_infos_[index].message_;
            if (message->err() == RdKafka::ERR_NO_ERROR) {
              // Produce the line as output.
              Tensor line_tensor(cpu_allocator(), DT_STRING, {});
              line_tensor.scalar<string>()() =
                  std::string(static_cast<const char*>(message->payload()),
                              message->len());
              out_tensors->emplace_back(std::move(line_tensor));
              if (dataset()->message_key_) {
                Tensor key_tensor(cpu_allocator(), DT_STRING, {});
                if (message->key() != nullptr) {
                  key_tensor.scalar<string>()() = string(*message->key());
                } else {
                  key_tensor.scalar<string>()() = "";
                }
                out_tensors->emplace_back(std::move(key_tensor));
              }
              *end_of_sequence = false;
              // Sync offset
              consumer_infos_[index].offset_ = message->offset();
              message.reset(nullptr);
              return Status::OK();
            }
          }
        }
        return errors::Internal(
            "Failed to consume due to all brokers down");
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        Tensor offset_tensor(DT_INT64, TensorShape({consumer_infos_.size()}));
        auto offset_t = offset_tensor.vec<int64>();
        for (int64 i = 0; i < consumer_infos_.size(); ++i) {
          offset_t(i) = consumer_infos_[i].offset_;
        }
        TF_RETURN_IF_ERROR(
            writer->WriteTensor(full_name("current_pos"), offset_tensor));
        LOG(INFO) << "Save all topic partition current offset." << offset_tensor.DebugString();
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        ResetStreamsLocked();
        if (reader->Contains(full_name("current_pos"))) {
          Tensor offset_tensor;
          TF_RETURN_IF_ERROR(
              reader->ReadTensor(full_name("current_pos"), &offset_tensor));
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
          auto offset_t = offset_tensor.vec<int64>();
          for (int64 i = 0; i < consumer_infos_.size(); ++i) {
            int64 current_pos = offset_t(i);
            consumer_infos_[i].topic_partition_->set_offset(current_pos);
            if (consumer_infos_[i].topic_partition_->offset() != current_pos) {
              return errors::Internal("Failed to restore to offset ",
                                      current_pos);
            }
            std::vector<RdKafka::TopicPartition*> partitions;
            partitions.emplace_back(consumer_infos_[i].topic_partition_.get());
            RdKafka::ErrorCode err = consumer_infos_[i].consumer_->assign(partitions);
            if (err != RdKafka::ERR_NO_ERROR) {
              return errors::Internal(
                  "Failed to assign partition:", RdKafka::err2str(err));
            }
          }
          LOG(INFO) << "Restore to topic partition all offset done.";
        }
        return Status::OK();
      }

     private:
      // Sets up Kafka streams
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        int32_t size = dataset()->topics_.size();
        consumer_infos_.resize(size);
        for (int32_t i = 0; i < size; ++i) {
          std::string entry = dataset()->topics_[i];

          std::vector<string> parts = str_util::Split(entry, ":");
          if (parts.size() < 1) {
            return errors::InvalidArgument("Invalid parameters: ", entry);
          }
          string topic = parts[0];
          int32 partition = 0;
          if (parts.size() > 1) {
            if (!strings::safe_strto32(parts[1], &partition)) {
              return errors::InvalidArgument("Invalid parameters: ", entry);
            }
          }
          int64 offset = 0;
          if (parts.size() > 2) {
            if (!strings::safe_strto64(parts[2], &offset)) {
              return errors::InvalidArgument("Invalid parameters: ", entry);
            }
          }

          consumer_infos_[i].topic_partition_.reset(
              RdKafka::TopicPartition::create(topic, partition, offset));

          consumer_infos_[i].offset_ = consumer_infos_[i].topic_partition_->offset();
          consumer_infos_[i].limit_ = -1;
          if (parts.size() > 3) {
            if (!strings::safe_strto64(parts[3], &consumer_infos_[i].limit_)) {
              return errors::InvalidArgument("Invalid parameters: ", entry);
            }
          }
        }

        std::unique_ptr<RdKafka::Conf> conf(
            RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
        std::unique_ptr<RdKafka::Conf> topic_conf(
            RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));
        RdKafka::Conf::ConfResult result = RdKafka::Conf::CONF_UNKNOWN;

        std::string errstr;

        for (auto it = dataset()->config_topic_.begin();
             it != dataset()->config_topic_.end(); it++) {
          std::vector<string> parts = str_util::Split(*it, "=");
          if (parts.size() != 2) {
            return errors::InvalidArgument("Invalid topic configuration: ",
                                           *it);
          }
          result = topic_conf->set(parts[0], parts[1], errstr);
          if (result != RdKafka::Conf::CONF_OK) {
            return errors::Internal("Failed to do topic configuration:", *it,
                                    "error:", errstr);
          }
          LOG(INFO) << "Kafka topic configuration: " << *it;
        }

        result = conf->set("default_topic_conf", topic_conf.get(), errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set default_topic_conf:", errstr);
        }

        for (auto it = dataset()->config_global_.begin();
             it != dataset()->config_global_.end(); it++) {
          std::vector<string> parts = str_util::Split(*it, "=");
          if (parts.size() != 2) {
            return errors::InvalidArgument("Invalid global configuration: ",
                                           *it);
          }
          result = conf->set(parts[0], parts[1], errstr);
          if (result != RdKafka::Conf::CONF_OK) {
            return errors::Internal("Failed to do global configuration: ", *it,
                                    "error:", errstr);
          }
          LOG(INFO) << "Kafka global configuration: " << *it;
        }

        result = conf->set("event_cb", &kafka_event_cb, errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set event_cb:", errstr);
        }

        result = conf->set("bootstrap.servers", dataset()->servers_, errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set bootstrap.servers ",
                                  dataset()->servers_, ":", errstr);
        }
        result = conf->set("group.id", dataset()->group_, errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set group.id ", dataset()->group_,
                                  ":", errstr);
        }

        // Always enable.partition.eof=true
        result = conf->set("enable.partition.eof", "true", errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("Failed to set enable.partition.eof=true",
                                  ":", errstr);
        }

        for (auto& iter : consumer_infos_) {
          iter.consumer_.reset(RdKafka::KafkaConsumer::create(conf.get(), errstr));
          if (!iter.consumer_.get()) {
            return errors::Internal("Failed to create consumer:", errstr);
          }
          std::vector<RdKafka::TopicPartition*> parts;
          parts.emplace_back(iter.topic_partition_.get());
          RdKafka::ErrorCode err = iter.consumer_->assign(parts);
          if (err != RdKafka::ERR_NO_ERROR) {
            return errors::Internal(
                "Failed to assign partition:", RdKafka::err2str(err));
          }
        }
        return Status::OK();
      }

      // Resets all Kafka streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        for (auto& iter : consumer_infos_) {
          auto& consumer = iter.consumer_;
          if (consumer.get()) {
            consumer->unassign();
            consumer.reset(nullptr);
          }
        }
      }

      mutex mu_;
      bool run_ GUARDED_BY(mu_) = true;
      struct ConsumerInfo {
        std::unique_ptr<RdKafka::TopicPartition> topic_partition_;
        std::unique_ptr<RdKafka::KafkaConsumer> consumer_;
        std::unique_ptr<RdKafka::Message> message_;
        int64 offset_ = 0;
        int64 limit_ = -1;
        bool eof_ = false;
      };
      bool init_ = false;
      std::vector<ConsumerInfo> consumer_infos_ GUARDED_BY(mu_);
      KafkaEventCb kafka_event_cb = KafkaEventCb(run_);
    };

    const std::vector<string> topics_;
    const string servers_;
    const string group_;
    const bool eof_;
    const int64 timeout_;
    const std::vector<string> config_global_;
    const std::vector<string> config_topic_;
    const bool message_key_;
  };
};

class WriteKafkaOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* context) override {
    const Tensor* message_tensor;
    const Tensor* topic_tensor;
    const Tensor* servers_tensor;
    OP_REQUIRES_OK(context, context->input("message", &message_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(message_tensor->shape()),
                errors::InvalidArgument(
                    "Message tensor must be scalar, but had shape: ",
                    message_tensor->shape().DebugString()));
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(topic_tensor->shape()),
        errors::InvalidArgument("Topic tensor must be scalar, but had shape: ",
                                topic_tensor->shape().DebugString()));
    OP_REQUIRES_OK(context, context->input("servers", &servers_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(servers_tensor->shape()),
                errors::InvalidArgument(
                    "Servers tensor must be scalar, but had shape: ",
                    servers_tensor->shape().DebugString()));

    const string& message = message_tensor->scalar<string>()();
    const string& topic_string = topic_tensor->scalar<string>()();
    std::vector<string> parts = str_util::Split(topic_string, ":");
    OP_REQUIRES(context, (parts.size() >= 1),
                errors::InvalidArgument("Invalid parameters: ", topic_string));

    const string& topic_str = parts[0];
    int32 partition = 0;
    if (parts.size() > 1) {
      OP_REQUIRES(
          context, !strings::safe_strto32(parts[1], &partition),
          errors::InvalidArgument("Invalid parameters: ", topic_string));
    }

    const string& servers = servers_tensor->scalar<string>()();

    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> topic_conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    std::string errstr;

    RdKafka::Conf::ConfResult result =
        conf->set("default_topic_conf", topic_conf.get(), errstr);
    OP_REQUIRES(context, (result == RdKafka::Conf::CONF_OK),
                errors::Internal("Failed to set default_topic_conf:", errstr));

    result = conf->set("bootstrap.servers", servers, errstr);
    OP_REQUIRES(context, (result == RdKafka::Conf::CONF_OK),
                errors::Internal("Failed to set bootstrap.servers ", servers,
                                 ":", errstr));

    std::unique_ptr<RdKafka::Producer> producer(
        RdKafka::Producer::create(conf.get(), errstr));
    OP_REQUIRES(context, producer.get() != nullptr,
                errors::Internal("Failed to create producer:", errstr));

    std::unique_ptr<RdKafka::Topic> topic(RdKafka::Topic::create(
        producer.get(), topic_str, topic_conf.get(), errstr));
    OP_REQUIRES(
        context, topic.get() != nullptr,
        errors::Internal("Failed to create topic ", topic_str, ":", errstr));

    RdKafka::ErrorCode err = producer->produce(
        topic.get(), partition, RdKafka::Producer::RK_MSG_COPY,
        const_cast<char*>(message.c_str()), message.size(), NULL, NULL);
    OP_REQUIRES(
        context, (err == RdKafka::ERR_NO_ERROR),
        errors::Internal("Failed to produce message:", RdKafka::err2str(err)));

    err = producer->flush(timeout_);
    OP_REQUIRES(
        context, (err == RdKafka::ERR_NO_ERROR),
        errors::Internal("Failed to flush message:", RdKafka::err2str(err)));
    context->set_output(0, context->input(0));
  }

 private:
  static const int timeout_ = 5000;
};

class KafkaEventCb : public RdKafka::EventCb {
 public:
  KafkaEventCb() : run_(true) {}

  bool run() { return run_; }

  void event_cb(RdKafka::Event& event) {
    switch (event.type()) {
      case RdKafka::Event::EVENT_ERROR:
        LOG(ERROR) << "EVENT_ERROR: "
                   << "(" << RdKafka::err2str(event.err())
                   << "): " << event.str();
        { run_ = !event.fatal(); }
        break;
      case RdKafka::Event::EVENT_STATS:
        LOG(ERROR) << "EVENT_STATS: " << event.str();
        break;
      case RdKafka::Event::EVENT_LOG:
        LOG(ERROR) << "EVENT_LOG: " << event.severity() << "-"
                   << event.fac().c_str() << "-" << event.str().c_str();
        break;
      case RdKafka::Event::EVENT_THROTTLE:
        LOG(ERROR) << "EVENT_THROTTLE: " << event.throttle_time() << "ms by "
                   << event.broker_name() << " id " << (int)event.broker_id();
        break;
      default:
        LOG(ERROR) << "EVENT: " << event.type() << " ("
                   << RdKafka::err2str(event.err()) << "): " << event.str();
        break;
    }
  }

 private:
  mutable mutex mu_;
  bool run_ TF_GUARDED_BY(mu_) = true;
};

class KafkaReadableResource : public ResourceBase {
 public:
  KafkaReadableResource(Env* env) : env_(env) {}
  virtual ~KafkaReadableResource() {
    if (consumer_.get()) {
      consumer_->unassign();
      consumer_->close();
      consumer_.reset(nullptr);
    }
  }

  virtual Status Init(const string& topic, const int32 partition,
                      const int64 offset, const std::vector<string>& metadata) {
    mutex_lock l(mu_);

    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> conf_topic(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    string errstr;
    RdKafka::Conf::ConfResult result = RdKafka::Conf::CONF_UNKNOWN;
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("conf.topic.") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid topic configuration: ",
                                         metadata[i]);
        }
        result = conf_topic->set(parts[0].substr(11), parts[1], errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do topic configuration:",
                                  metadata[i], "error:", errstr);
        }
      } else if (metadata[i] != "" &&
                 metadata[i].find("conf.") == string::npos) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid topic configuration: ",
                                         metadata[i]);
        }
        if ((result = conf->set(parts[0], parts[1], errstr)) !=
            RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do global configuration: ",
                                  metadata[i], "error:", errstr);
        }
      }
      LOG(INFO) << "Kafka configuration: " << metadata[i];
    }
    if ((result = conf->set("default_topic_conf", conf_topic.get(), errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set default_topic_conf:", errstr);
    }

    // consumer.properties:
    //   bootstrap.servers=localhost:9092
    //   group.id=test-consumer-group
    string bootstrap_servers;
    if ((result = conf->get("bootstrap.servers", bootstrap_servers)) !=
        RdKafka::Conf::CONF_OK) {
      bootstrap_servers = "localhost:9092";
      if ((result = conf->set("bootstrap.servers", bootstrap_servers,
                              errstr)) != RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set bootstrap.servers [",
                                bootstrap_servers, "]:", errstr);
      }
    }
    string group_id;
    if ((result = conf->get("group.id", group_id)) != RdKafka::Conf::CONF_OK) {
      group_id = "test-consumer-group";
      if ((result = conf->set("group.id", group_id, errstr)) !=
          RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set group.id [", group_id,
                                "]:", errstr);
      }
    }

    // Always set enable.partition.eof=true
    if ((result = conf->set("enable.partition.eof", "true", errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("Failed to set enable.partition.eof=true :",
                              errstr);
    }

    if ((result = conf->set("event_cb", &kafka_event_cb_, errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set event_cb:", errstr);
    }

    consumer_.reset(RdKafka::KafkaConsumer::create(conf.get(), errstr));
    if (!consumer_.get()) {
      return errors::Internal("failed to create consumer:", errstr);
    }

    subscription_.reset(RdKafka::TopicPartition::create(topic, partition));
    std::vector<RdKafka::TopicPartition*> partitions;
    partitions.emplace_back(subscription_.get());

    subscription_->set_offset(offset);
    RdKafka::ErrorCode err = consumer_->assign(partitions);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to assign partition: ",
                              RdKafka::err2str(err));
    }

    return Status::OK();
  }
  Status Next(const int64 index,
              std::function<Status(const TensorShape& shape, Tensor** message,
                                   Tensor** key)>
                  allocate_func) {
    mutex_lock l(mu_);
    int64 total = 1024;
    std::vector<string> message_value, key_value;
    message_value.reserve(total);
    key_value.reserve(total);

    LOG(INFO) << "Kafka stream starts with current offset: "
              << subscription_->offset();
    std::unique_ptr<RdKafka::Message> message;
    int64 count = 0;
    while (consumer_.get() != nullptr && count < total) {
      if (!kafka_event_cb_.run()) {
        return errors::Internal("failed to consume due to all brokers down");
      }
      message.reset(consumer_->consume(timeout_));
      if (message->err() == RdKafka::ERR_NO_ERROR) {
        // Produce the line as output.
        message_value.emplace_back(string(
            static_cast<const char*>(message->payload()), message->len()));
        key_value.emplace_back(
            (message->key() != nullptr) ? string(*message->key()) : "");
        count++;
        continue;
      } else if (message->err() == RdKafka::ERR__TRANSPORT) {
        // Not return error here because consumer will try re-connect.
        LOG(ERROR) << "Broker transport failure: " << message->errstr();
      } else if (message->err() == RdKafka::ERR__PARTITION_EOF) {
        LOG(ERROR) << "EOF Message: " << message->errstr();
        consumer_.reset(nullptr);
        break;
      } else if (message->err() != RdKafka::ERR__TIMED_OUT) {
        LOG(ERROR) << "Failed to consume: " << message->errstr();
        return errors::Internal("Failed to consume: ", message->errstr());
      }
    }
    TensorShape shape({static_cast<int64>(message_value.size())});
    Tensor* message_tensor;
    Tensor* key_tensor;
    TF_RETURN_IF_ERROR(allocate_func(shape, &message_tensor, &key_tensor));
    for (size_t i = 0; i < message_value.size(); i++) {
      message_tensor->flat<tstring>()(i) = message_value[i];
      key_tensor->flat<tstring>()(i) = key_value[i];
    }
    return Status::OK();
  }
  Status Read(const int64 start, const int64 stop,
              std::function<Status(const TensorShape& shape, Tensor** message,
                                   Tensor** key)>
                  allocate_func) {
    mutex_lock l(mu_);

    int64 stop_offset;
    if (stop >= 0) {
      stop_offset = stop;
    } else if (stop == RdKafka::Topic::OFFSET_END) {
      stop_offset = RdKafka::Consumer::OffsetTail(0);
    } else if (stop <= RdKafka::Consumer::OffsetTail(0)) {
      stop_offset = stop;
    } else {
      return errors::InvalidArgument("stop offset ", stop, " not supported");
    }
    if (stop_offset <= RdKafka::Consumer::OffsetTail(0)) {
      int64 tail_offset = 0;
      TF_RETURN_IF_ERROR(Tail(&tail_offset));

      stop_offset =
          tail_offset + stop_offset - RdKafka::Consumer::OffsetTail(0);
    }

    std::vector<string> message_value, key_value;

    subscription_->set_offset(start);
    RdKafka::ErrorCode err = consumer_->seek((*subscription_), timeout_);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to seek partition: ",
                              RdKafka::err2str(err));
    }
    LOG(INFO) << "Kafka stream starts with current offset: "
              << subscription_->offset();
    int64 index = start;
    std::unique_ptr<RdKafka::Message> message;
    while (consumer_.get() != nullptr && index + 1 < stop_offset) {
      if (!kafka_event_cb_.run()) {
        return errors::Internal("failed to consume due to all brokers down");
      }
      message.reset(consumer_->consume(timeout_));
      if (message->err() == RdKafka::ERR_NO_ERROR) {
        // Produce the line as output.
        message_value.emplace_back(string(
            static_cast<const char*>(message->payload()), message->len()));
        key_value.emplace_back(
            (message->key() != nullptr) ? string(*message->key()) : "");
        index = message->offset();
        continue;
      } else if (message->err() == RdKafka::ERR__PARTITION_EOF) {
        LOG(ERROR) << "EOF Message: " << message->errstr();
        break;
      } else if (message->err() == RdKafka::ERR__TRANSPORT) {
        // Not return error here because consumer will try re-connect.
        LOG(ERROR) << "Broker transport failure: " << message->errstr();
      } else if (message->err() != RdKafka::ERR__TIMED_OUT) {
        LOG(ERROR) << "Failed to consume: " << message->errstr();
        return errors::Internal("Failed to consume: ", message->errstr());
      }
    }
    TensorShape shape({static_cast<int64>(message_value.size())});
    Tensor* message_tensor;
    Tensor* key_tensor;
    TF_RETURN_IF_ERROR(allocate_func(shape, &message_tensor, &key_tensor));
    for (size_t i = 0; i < message_value.size(); i++) {
      message_tensor->flat<tstring>()(i) = message_value[i];
      key_tensor->flat<tstring>()(i) = key_value[i];
    }
    return Status::OK();
  }
  Status Spec(const int64 start, const int64 stop, int64* start_offset,
              int64* stop_offset) {
    mutex_lock l(mu_);

    if (start >= 0) {
      *start_offset = start;
    } else if (start == RdKafka::Topic::OFFSET_END) {
      *start_offset = RdKafka::Consumer::OffsetTail(0);
    } else if (start <= RdKafka::Consumer::OffsetTail(0)) {
      *start_offset = start;
    } else {
      return errors::InvalidArgument("start offset ", start, " not supported");
    }

    if (stop >= 0) {
      *stop_offset = stop;
    } else if (stop == RdKafka::Topic::OFFSET_END) {
      *stop_offset = RdKafka::Consumer::OffsetTail(0);
    } else if (stop <= RdKafka::Consumer::OffsetTail(0)) {
      *stop_offset = stop;
    } else {
      return errors::InvalidArgument("stop offset ", stop, " not supported");
    }

    if (*start_offset <= RdKafka::Consumer::OffsetTail(0) ||
        *stop_offset <= RdKafka::Consumer::OffsetTail(0)) {
      int64 tail_offset = 0;
      TF_RETURN_IF_ERROR(Tail(&tail_offset));

      if (*start_offset <= RdKafka::Consumer::OffsetTail(0)) {
        *start_offset =
            tail_offset + *start_offset - RdKafka::Consumer::OffsetTail(0);
      }
      if (*stop_offset <= RdKafka::Consumer::OffsetTail(0)) {
        *stop_offset =
            tail_offset + *stop_offset - RdKafka::Consumer::OffsetTail(0);
      }
    }

    return Status::OK();
  }
  string DebugString() const override { return "KafkaBaseResource"; }

 protected:
  Status Tail(int64* tail_offset) {
    // Resolve tail message
    int64 saved = subscription_->offset();

    subscription_->set_offset(RdKafka::Consumer::OffsetTail(1));
    RdKafka::ErrorCode err = consumer_->seek(*subscription_, timeout_);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to seek tail -1: ",
                              RdKafka::err2str(err));
    }
    std::unique_ptr<RdKafka::Message> message;
    do {
      message.reset(consumer_->consume(timeout_));
    } while (message->err() == RdKafka::ERR__TRANSPORT);
    if (message->err() != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to consume tail message: ",
                              RdKafka::err2str(message->err()));
    }
    *tail_offset = message->offset() + 1;
    LOG(INFO) << "Kafka tail: " << *tail_offset;

    subscription_->set_offset(saved);
    err = consumer_->seek(*subscription_, timeout_);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to seek back saved: ",
                              RdKafka::err2str(err));
    }

    return Status::OK();
  }
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::TopicPartition> subscription_ TF_GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::KafkaConsumer> consumer_ TF_GUARDED_BY(mu_);
  KafkaEventCb kafka_event_cb_ = KafkaEventCb();
  static const int timeout_ = 5000;
};

class KafkaReadableInitOp : public ResourceOpKernel<KafkaReadableResource> {
 public:
  explicit KafkaReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<KafkaReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<KafkaReadableResource>::Compute(context);

    const Tensor* topic_tensor;
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));
    const string& topic = topic_tensor->scalar<tstring>()();

    const Tensor* partition_tensor;
    OP_REQUIRES_OK(context, context->input("partition", &partition_tensor));
    const int32 partition = partition_tensor->scalar<int32>()();

    const Tensor* offset_tensor;
    OP_REQUIRES_OK(context, context->input("offset", &offset_tensor));
    const int64 offset = offset_tensor->scalar<int64>()();

    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    std::vector<string> metadata;
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<tstring>()(i));
    }

    OP_REQUIRES_OK(context,
                   resource_->Init(topic, partition, offset, metadata));
  }
  Status CreateResource(KafkaReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new KafkaReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class KafkaReadableNextOp : public OpKernel {
 public:
  explicit KafkaReadableNextOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    KafkaReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));
    const int64 index = index_tensor->scalar<int64>()();

    OP_REQUIRES_OK(
        context,
        resource->Next(
            index,
            [&](const TensorShape& shape, Tensor** message,
                Tensor** key) -> Status {
              TF_RETURN_IF_ERROR(context->allocate_output(0, shape, message));
              TF_RETURN_IF_ERROR(context->allocate_output(1, shape, key));
              return Status::OK();
            }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class KafkaReadableReadOp : public OpKernel {
 public:
  explicit KafkaReadableReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    KafkaReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    const int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    const int64 stop = stop_tensor->scalar<int64>()();

    OP_REQUIRES_OK(
        context,
        resource->Read(
            start, stop,
            [&](const TensorShape& shape, Tensor** message,
                Tensor** key) -> Status {
              TF_RETURN_IF_ERROR(context->allocate_output(0, shape, message));
              TF_RETURN_IF_ERROR(context->allocate_output(1, shape, key));
              return Status::OK();
            }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class KafkaReadableSpecOp : public OpKernel {
 public:
  explicit KafkaReadableSpecOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    KafkaReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* start_tensor;
    OP_REQUIRES_OK(context, context->input("start", &start_tensor));
    const int64 start = start_tensor->scalar<int64>()();

    const Tensor* stop_tensor;
    OP_REQUIRES_OK(context, context->input("stop", &stop_tensor));
    const int64 stop = stop_tensor->scalar<int64>()();

    int64 start_offset, stop_offset;
    OP_REQUIRES_OK(context,
                   resource->Spec(start, stop, &start_offset, &stop_offset));

    Tensor* start_offset_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                     &start_offset_tensor));
    start_offset_tensor->scalar<int64>()() = start_offset;

    Tensor* stop_offset_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &stop_offset_tensor));
    stop_offset_tensor->scalar<int64>()() = stop_offset;
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class LayerKafkaResource : public ResourceBase {
 public:
  LayerKafkaResource(Env* env) : env_(env) {}
  ~LayerKafkaResource() { Sync(); }

  Status Init(const string& topic, const int32 partition,
              const std::vector<string>& metadata) {
    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> conf_topic(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    string errstr;
    RdKafka::Conf::ConfResult result = RdKafka::Conf::CONF_UNKNOWN;

    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("conf.topic.") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid topic configuration: ",
                                         metadata[i]);
        }
        result = conf_topic->set(parts[0].substr(11), parts[1], errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do topic configuration:",
                                  metadata[i], "error:", errstr);
        }
      } else if (metadata[i] != "" &&
                 metadata[i].find("conf.") == string::npos) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid global configuration: ",
                                         metadata[i]);
        }
        if ((result = conf->set(parts[0], parts[1], errstr)) !=
            RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do global configuration: ",
                                  metadata[i], "error:", errstr);
        }
      }
      LOG(INFO) << "Kafka configuration: " << metadata[i];
    }
    if ((result = conf->set("default_topic_conf", conf_topic.get(), errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set default_topic_conf:", errstr);
    }

    string bootstrap_servers;
    if ((result = conf->get("bootstrap.servers", bootstrap_servers)) !=
        RdKafka::Conf::CONF_OK) {
      bootstrap_servers = "localhost:9092";
      if ((result = conf->set("bootstrap.servers", bootstrap_servers,
                              errstr)) != RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set bootstrap.servers [",
                                bootstrap_servers, "]:", errstr);
      }
      LOG(INFO) << "Kafka default bootstrap server: " << bootstrap_servers;
    }

    producer_.reset(RdKafka::Producer::create(conf.get(), errstr));
    if (!(producer_.get() != nullptr)) {
      return errors::Internal("Failed to create producer:", errstr);
    }

    topic_.reset(RdKafka::Topic::create(producer_.get(), topic,
                                        conf_topic.get(), errstr));
    if (!(topic_.get() != nullptr)) {
      return errors::Internal("Failed to create topic ", topic, ":", errstr);
    }

    partition_ = partition;
    return Status::OK();
  }
  Status Write(const Tensor& content) {
    mutex_lock l(mu_);
    for (int64 i = 0; i < content.NumElements(); i++) {
      RdKafka::ErrorCode err = producer_->produce(
          topic_.get(), partition_, RdKafka::Producer::RK_MSG_COPY,
          (void*)content.flat<tstring>()(i).data(),
          content.flat<tstring>()(i).size(), NULL, NULL);
      if (!(err == RdKafka::ERR_NO_ERROR)) {
        return errors::Internal("Failed to produce message:",
                                RdKafka::err2str(err));
      }
    }
    return Status::OK();
  }
  Status Sync() {
    if (producer_.get() != nullptr) {
      RdKafka::ErrorCode err = producer_->flush(timeout_);
      if (!(err == RdKafka::ERR_NO_ERROR)) {
        return errors::Internal("Failed to flush message:",
                                RdKafka::err2str(err));
      }
    }
    return Status::OK();
  }
  string DebugString() const override { return "LayerKafkaResource"; }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::Producer> producer_ TF_GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::Topic> topic_ TF_GUARDED_BY(mu_);
  int32 partition_ TF_GUARDED_BY(mu_);
  static const int timeout_ = 5000;
};

class LayerKafkaInitOp : public ResourceOpKernel<LayerKafkaResource> {
 public:
  explicit LayerKafkaInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<LayerKafkaResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<LayerKafkaResource>::Compute(context);

    const Tensor* topic_tensor;
    OP_REQUIRES_OK(context, context->input("topic", &topic_tensor));

    const Tensor* partition_tensor;
    OP_REQUIRES_OK(context, context->input("partition", &partition_tensor));

    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    std::vector<string> metadata;
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<tstring>()(i));
    }

    OP_REQUIRES_OK(context, resource_->Init(topic_tensor->scalar<tstring>()(),
                                            partition_tensor->scalar<int32>()(),
                                            metadata));
  }
  Status CreateResource(LayerKafkaResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new LayerKafkaResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class LayerKafkaCallOp : public OpKernel {
 public:
  explicit LayerKafkaCallOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }

    const Tensor* content_tensor;
    OP_REQUIRES_OK(context, context->input("content", &content_tensor));

    LayerKafkaResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    OP_REQUIRES_OK(context, resource->Write(*content_tensor));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class LayerKafkaSyncOp : public OpKernel {
 public:
  explicit LayerKafkaSyncOp(OpKernelConstruction* context) : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    LayerKafkaResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);
    OP_REQUIRES_OK(context, resource->Sync());
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

static int64 partition_count = 0;
static int64 eof_count = 0;
class KafkaRebalanceCb : public RdKafka::RebalanceCb {
 public:
  KafkaRebalanceCb() : run_(true) {}

  bool run() { return run_; }

  void rebalance_cb(RdKafka::KafkaConsumer* consumer, RdKafka::ErrorCode err,
                    std::vector<RdKafka::TopicPartition*>& partitions) {
    LOG(ERROR) << "REBALANCE: " << RdKafka::err2str(err);
    int timeout = 5000;  // milliseconds
    LOG(ERROR) << "Retrieved committed offsets with status code: "
               << consumer->committed(partitions, timeout);

    for (int partition = 0; partition < partitions.size(); partition++) {
      // OFFSET MAPPINGS:
      //
      // RD_KAFKA_OFFSET_BEGINNING      -2
      // RD_KAFKA_OFFSET_END            -1
      // RD_KAFKA_OFFSET_STORED         -1000
      // RD_KAFKA_OFFSET_INVALID        -1001

      LOG(INFO) << "REBALANCE: " << partitions[partition]->topic() << "["
                << partitions[partition]->partition() << "], "
                << "OFFSET: " << partitions[partition]->offset() << " "
                << "ERROR_CODE: " << partitions[partition]->err();
    }
    if (err == RdKafka::ERR__ASSIGN_PARTITIONS) {
      // librdkafka does not actually look up the stored offsets before
      // calling your rebalance callback, the partition offsets are set to
      // RD_KAFKA_OFFSET_INVALID at this point to allow us to change it to use
      // some sort of external offset store. But calling assign() with offset
      // RD_KAFKA_OFFSET_INVALID will cause librdkafka to look up the stored
      // offset on the broker.
      // If there was no stored offset it will fall back to `auto.offset.reset`
      // configuration parameter.

      LOG(INFO) << "REBALANCE: Assigning partitions";
      consumer->assign(partitions);
      partition_count = (int)partitions.size();
    } else {
      LOG(INFO) << "REBALANCE: Unassigning partitions";
      consumer->unassign();
      partition_count = 0;
    }
    eof_count = 0;
  }

 private:
  mutable mutex mu_;
  bool run_ TF_GUARDED_BY(mu_) = true;
};

class KafkaGroupReadableResource : public ResourceBase {
 public:
  KafkaGroupReadableResource(Env* env) : env_(env) {}
  virtual ~KafkaGroupReadableResource() {
    if (consumer_.get()) {
      consumer_->unassign();
      consumer_->close();
      consumer_.reset(nullptr);
    }
  }

  virtual Status Init(const std::vector<std::string>& topics,
                      const std::vector<std::string>& metadata) {
    mutex_lock l(mu_);

    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> conf_topic(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    string errstr;
    RdKafka::Conf::ConfResult result = RdKafka::Conf::CONF_UNKNOWN;

    // The default kafka topic configurations are set first before
    // setting the global confs
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i].find("conf.topic.") == 0) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid topic configuration: ",
                                         metadata[i]);
        }
        result = conf_topic->set(parts[0].substr(11), parts[1], errstr);
        if (result != RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do topic configuration:",
                                  metadata[i], "error:", errstr);
        }
        LOG(INFO) << "Kafka configuration: " << metadata[i];
      }
    }
    if ((result = conf->set("default_topic_conf", conf_topic.get(), errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set default_topic_conf:", errstr);
    }

    // Once the `default_topic_conf` is set, the global confs can now be set
    // without any risk of being overwritten.
    // Setting the global confs before setting the `default_topic_conf`
    // results in erratic behaviour.
    for (size_t i = 0; i < metadata.size(); i++) {
      if (metadata[i] != "" && metadata[i].find("conf.") == string::npos) {
        std::vector<string> parts = str_util::Split(metadata[i], "=");
        if (parts.size() != 2) {
          return errors::InvalidArgument("invalid topic configuration: ",
                                         metadata[i]);
        }
        if ((result = conf->set(parts[0], parts[1], errstr)) !=
            RdKafka::Conf::CONF_OK) {
          return errors::Internal("failed to do global configuration: ",
                                  metadata[i], "error:", errstr);
        }
        LOG(INFO) << "Kafka configuration: " << metadata[i];
      }
    }

    // default consumer.properties:
    //   bootstrap.servers=localhost:9092
    //   group.id=test-consumer-group

    string bootstrap_servers;
    if ((result = conf->get("bootstrap.servers", bootstrap_servers)) !=
        RdKafka::Conf::CONF_OK) {
      bootstrap_servers = "localhost:9092";
      if ((result = conf->set("bootstrap.servers", bootstrap_servers,
                              errstr)) != RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set bootstrap.servers [",
                                bootstrap_servers, "]:", errstr);
      }
    }
    string group_id;
    if ((result = conf->get("group.id", group_id)) != RdKafka::Conf::CONF_OK) {
      group_id = "test-consumer-group";
      if ((result = conf->set("group.id", group_id, errstr)) !=
          RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set group.id [", group_id,
                                "]:", errstr);
      }
    }

    // set max.poll.records configuration
    std::string batch_num_messages;
    if ((result = conf->get("batch.num.messages", batch_num_messages)) !=
        RdKafka::Conf::CONF_OK) {
      batch_num_messages = "1024";
      if ((result = conf->set("batch.num.messages", batch_num_messages,
                              errstr)) != RdKafka::Conf::CONF_OK) {
        return errors::Internal("failed to set batch.num.messages [",
                                batch_num_messages, "]:", errstr);
      }
    }
    sscanf(batch_num_messages.c_str(), "%d", &batch_num_messages_);
    LOG(INFO) << "max num of messages per batch: " << batch_num_messages_;

    // Always set enable.partition.eof=true
    if ((result = conf->set("enable.partition.eof", "true", errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("Failed to set enable.partition.eof=true :",
                              errstr);
    }

    if ((result = conf->set("event_cb", &kafka_event_cb_, errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set event_cb:", errstr);
    }

    if ((result = conf->set("rebalance_cb", &kafka_rebalance_cb_, errstr)) !=
        RdKafka::Conf::CONF_OK) {
      return errors::Internal("failed to set rebalance_cb:", errstr);
    }

    LOG(INFO) << "Creating the kafka consumer";
    consumer_.reset(RdKafka::KafkaConsumer::create(conf.get(), errstr));
    if (!consumer_.get()) {
      return errors::Internal("failed to create consumer:", errstr);
    }

    for (int i = 0; i < topics.size(); i++) {
      LOG(INFO) << "Subscribing to the kafka topic: " << topics[i];
    }
    RdKafka::ErrorCode err = consumer_->subscribe(topics);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("failed to subscribe to topics: ",
                              RdKafka::err2str(err));
    }

    return Status::OK();
  }
  Status Next(const int64 index, const int64 message_poll_timeout,
              const int64 stream_timeout,
              std::function<Status(const TensorShape& shape, Tensor** message,
                                   Tensor** key, Tensor** continue_fetch)>
                  allocate_func) {
    mutex_lock l(mu_);

    // Initialize necessary variables
    int64 num_messages = 0;
    max_stream_timeout_polls_ = stream_timeout / message_poll_timeout;

    // Allocate memory for message_value and key_value vectors
    std::vector<string> message_value, key_value;
    message_value.reserve(batch_num_messages_);
    key_value.reserve(batch_num_messages_);

    std::unique_ptr<RdKafka::Message> message;
    while (consumer_.get() != nullptr && num_messages < batch_num_messages_) {
      if (!kafka_event_cb_.run()) {
        return errors::Internal(
            "failed to consume messages due to broker issue");
      }
      message.reset(consumer_->consume(message_poll_timeout));
      if (message->err() == RdKafka::ERR_NO_ERROR) {
        // Produce the line as output.
        message_value.emplace_back(string(
            static_cast<const char*>(message->payload()), message->len()));
        key_value.emplace_back(
            (message->key() != nullptr) ? string(*message->key()) : "");
        num_messages++;
        // Once a message has been successfully retrieved, the
        // `stream_timeout_polls_` is reset to 0. This allows the dataset
        // to wait for the entire `stream_timeout` duration when a data
        // slump occurs in the future.
        stream_timeout_polls_ = 0;
      } else if (message->err() == RdKafka::ERR__TRANSPORT) {
        // Not returning an error here as the consumer will try to re-connect.
        LOG(ERROR) << "Broker transport failure: " << message->errstr();

      } else if (message->err() == RdKafka::ERR__PARTITION_EOF) {
        if (++eof_count == partition_count) {
          LOG(INFO) << "EOF reached for all " << partition_count
                    << " partition(s)";
          break;
        }
      } else if (message->err() == RdKafka::ERR__TIMED_OUT) {
        LOG(ERROR) << message->errstr();
        stream_timeout_polls_++;
        break;
      }
    }

    // Prepare the outputs
    TensorShape shape({static_cast<int64>(message_value.size())});
    Tensor* message_tensor;
    Tensor* key_tensor;
    Tensor* continue_fetch_tensor;
    TF_RETURN_IF_ERROR(allocate_func(shape, &message_tensor, &key_tensor,
                                     &continue_fetch_tensor));

    if (stream_timeout_polls_ < max_stream_timeout_polls_) {
      continue_fetch_tensor->scalar<int64>()() = 1;
    } else {
      continue_fetch_tensor->scalar<int64>()() = 0;
    }
    for (size_t i = 0; i < message_value.size(); i++) {
      message_tensor->flat<tstring>()(i) = message_value[i];
      key_tensor->flat<tstring>()(i) = key_value[i];
    }

    return Status::OK();
  }

  string DebugString() const override { return "KafkaBaseResource"; }

  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
  // std::unique_ptr<RdKafka::TopicPartition> subscription_ TF_GUARDED_BY(mu_);
  std::unique_ptr<RdKafka::KafkaConsumer> consumer_ TF_GUARDED_BY(mu_);
  KafkaEventCb kafka_event_cb_ = KafkaEventCb();
  KafkaRebalanceCb kafka_rebalance_cb_ = KafkaRebalanceCb();
  int max_stream_timeout_polls_ = -1;
  int stream_timeout_polls_ = -1;
  int batch_num_messages_ = 1024;
};

class KafkaGroupReadableInitOp
    : public ResourceOpKernel<KafkaGroupReadableResource> {
 public:
  explicit KafkaGroupReadableInitOp(OpKernelConstruction* context)
      : ResourceOpKernel<KafkaGroupReadableResource>(context) {
    env_ = context->env();
  }

 private:
  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<KafkaGroupReadableResource>::Compute(context);

    const Tensor* topics_tensor;
    OP_REQUIRES_OK(context, context->input("topics", &topics_tensor));
    std::vector<string> topics;
    for (int64 i = 0; i < topics_tensor->NumElements(); i++) {
      topics.push_back(topics_tensor->flat<tstring>()(i));
    }

    const Tensor* metadata_tensor;
    OP_REQUIRES_OK(context, context->input("metadata", &metadata_tensor));
    std::vector<string> metadata;
    for (int64 i = 0; i < metadata_tensor->NumElements(); i++) {
      metadata.push_back(metadata_tensor->flat<tstring>()(i));
    }

    OP_REQUIRES_OK(context, resource_->Init(topics, metadata));
  }
  Status CreateResource(KafkaGroupReadableResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new KafkaGroupReadableResource(env_);
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

class KafkaGroupReadableNextOp : public OpKernel {
 public:
  explicit KafkaGroupReadableNextOp(OpKernelConstruction* context)
      : OpKernel(context) {
    env_ = context->env();
  }

  void Compute(OpKernelContext* context) override {
    KafkaGroupReadableResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "input", &resource));
    core::ScopedUnref unref(resource);

    const Tensor* index_tensor;
    OP_REQUIRES_OK(context, context->input("index", &index_tensor));
    const int64 index = index_tensor->scalar<int64>()();

    const Tensor* message_poll_timeout_tensor;
    OP_REQUIRES_OK(context, context->input("message_poll_timeout",
                                           &message_poll_timeout_tensor));
    const int64 message_poll_timeout =
        message_poll_timeout_tensor->scalar<int64>()();

    const Tensor* stream_timeout_tensor;
    OP_REQUIRES_OK(context,
                   context->input("stream_timeout", &stream_timeout_tensor));
    const int64 stream_timeout = stream_timeout_tensor->scalar<int64>()();

    OP_REQUIRES_OK(
        context,
        resource->Next(
            index, message_poll_timeout, stream_timeout,
            [&](const TensorShape& shape, Tensor** message, Tensor** key,
                Tensor** continue_fetch) -> Status {
              TF_RETURN_IF_ERROR(context->allocate_output(0, shape, message));
              TF_RETURN_IF_ERROR(context->allocate_output(1, shape, key));
              TF_RETURN_IF_ERROR(
                  context->allocate_output(2, TensorShape({}), continue_fetch));
              return Status::OK();
            }));
  }

 private:
  mutable mutex mu_;
  Env* env_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("IOKafkaDataset").Device(DEVICE_CPU),
                        KafkaDatasetOp);

REGISTER_KERNEL_BUILDER(Name("IOWriteKafka").Device(DEVICE_CPU), WriteKafkaOp);

REGISTER_KERNEL_BUILDER(Name("IOKafkaReadableInit").Device(DEVICE_CPU),
                        KafkaReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IOKafkaReadableNext").Device(DEVICE_CPU),
                        KafkaReadableNextOp);
REGISTER_KERNEL_BUILDER(Name("IOKafkaReadableRead").Device(DEVICE_CPU),
                        KafkaReadableReadOp);
REGISTER_KERNEL_BUILDER(Name("IOKafkaReadableSpec").Device(DEVICE_CPU),
                        KafkaReadableSpecOp);
REGISTER_KERNEL_BUILDER(Name("IOLayerKafkaInit").Device(DEVICE_CPU),
                        LayerKafkaInitOp);
REGISTER_KERNEL_BUILDER(Name("IOLayerKafkaCall").Device(DEVICE_CPU),
                        LayerKafkaCallOp);
REGISTER_KERNEL_BUILDER(Name("IOLayerKafkaSync").Device(DEVICE_CPU),
                        LayerKafkaSyncOp);
REGISTER_KERNEL_BUILDER(Name("IOKafkaGroupReadableInit").Device(DEVICE_CPU),
                        KafkaGroupReadableInitOp);
REGISTER_KERNEL_BUILDER(Name("IOKafkaGroupReadableNext").Device(DEVICE_CPU),
                        KafkaGroupReadableNextOp);

}  // namespace data

}  // namespace tensorflow

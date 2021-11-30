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
        { run_ = (event.err() != RdKafka::ERR__ALL_BROKERS_DOWN); }
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
  bool run_ GUARDED_BY(mu_) = true;
};

REGISTER_KERNEL_BUILDER(Name("IOKafkaDataset").Device(DEVICE_CPU),
                        KafkaDatasetOp);

REGISTER_KERNEL_BUILDER(Name("IOWriteKafka").Device(DEVICE_CPU), WriteKafkaOp);

}  // namespace data

}  // namespace tensorflow

// Copyright (c) 2017, Alibaba Inc.
// All right reserved.
//
// Author: Cao Zongyan <zongyan.cao@alibaba-inc.com>
// Created: 2017/09/15
//
// Description
//     Decode and transfer an input CSV format tensor into a specified tensor.
//
//     TransCsvID2SparseOp: normal CSV index string to Sparse tensor.
//             input arguments: "records", "max_id", "default_value",
//             attribute arguments: "id_as_value", "field_delim".
//         example:
//             Input: records(["2,10","7","0,8"]), max_id(12), default_value(1)
//             Attr: id_as_value(True), field_delim(',')
//             Output: indices([[0,2],[0,10],[1,7],[2,0],[2,8]]),
//                     values([2,10,7,0,8]),
//                     dense_shape([3,12])
//
//     TransCsvID2DenseOp: normal CSV index string to Dense matrix tensor.
//             input arguments: "records", "max_id", "default_value",
//             attribute arguments: "id_as_value", "field_delim".
//         example:
//             Input: records(["2,1","3","0,2"]), max_id(4), default_value(1)
//             Attr: id_as_value(False), field_delim(',')
//             Output: [[0,1,1,0],[0,0,0,1],[1,0,1,0]]
//
//     TransCsvKV2SparseOp: kv format CSV string to Sparse tensor.
//             input arguments: "records", "max_id"
//             attribute arguments: "T", "field_delim".
//         example:
//             Input: records(["2:2.0,10:0.1","7:-0.7","8:0.8"]), max_id(12),
//             Attr: T(DT_FLOAT32), field_delim(',')
//             Output: indices([[0,2],[0,10],[1,7],[2,0],[2,8]]),
//                     values([2.0,0.1,-0.7,0.8]),
//                     dense_shape([3,12])
//
//     TransCsvKV2DenseOp: kv format CSV string to Dense matrix tensor.
//             input arguments: "records", "max_id"
//             attribute arguments: "T", "field_delim".
//         example:
//             Input: records(["2:0.2,1:0.1","3:-0.3","0:0.4,2:0.2"]), max_id(4)
//             Attr: T(DT_FLOAT32), field_delim(',')
//             Output: [[0.0,0.1,0.2,0.0],[0.0,0.0,0.0,-0.3],[0.4,0.0,0.2,0.0]]
//
//     TransCsvToDenseOp: normal CSV number string to Dense matrix tensor.
//             input arguments: "records", "max_id"
//             attribute arguments: "T", "field_delim".
//         example:
//             Input: records(["0.2,0.1","-0.3","0.4,0.2"]), max_id(4)
//             Attr: T(DT_FLOAT32), field_delim(',')
//             Output: [[0.1,0.2,0.0,0.0],[-0.3,0.0,0.0,0.0],[0.4,0.2,0.0,0.0]]
//

#include <algorithm>
#include <cmath>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/util/work_sharder.h"

#define OP_CHECK_STATUS(OBJ, MU, EXP, STATUS) \
if (!TF_PREDICT_TRUE(EXP)) {                  \
  mutex_lock l((MU));                         \
  (OBJ) = (STATUS);                           \
  return;                                     \
}

namespace tensorflow {

#define ID_AUTO_DETECT_TAG -1048575 // hacking for auto detect column size

namespace {

bool DelimSupported(const char delim) {
    // Illegal delims are defined here.
    // These chars would be used in numbers.
    const StringPiece illegal_delims("0123456789+-Ee.");
    return (delim != '\0' &&
            illegal_delims.find(delim) == StringPiece::npos);
}

inline bool GetValueOnTheFly(const char* input, size_t input_size, size_t* pos, int64* value)
{
  // on-the-fly strtoi processing.
  // No overflow check here due to performance reason.
  int64 sign = 1;
  if (input[*pos] == '-') {
    sign = -1;
    ++(*pos);
  } else if (input[*pos] == '+') {
    ++(*pos);
  }
  *value = 0;
  if (input[*pos] >= '0' && input[*pos] <= '9') {
    while (*pos < input_size && input[*pos] >= '0' && input[*pos] <= '9') {
      (*value) = (*value) * 10 + input[*pos] - '0';
      ++(*pos);
    }
    (*value) *= sign;
    return true;
  }
  return false;
}

inline bool GetValueOnTheFly(const char* input, size_t input_size, size_t* pos, int32* value)
{
  int64 v0;
  if (GetValueOnTheFly(input, input_size, pos, &v0)) {
    // No overflow check here due to performance reason.
    *value = static_cast<int32>(v0);
    return true;
  }
  return false;
}

inline bool GetValueOnTheFly(const char* input, size_t input_size, size_t* pos, float* value)
{
  // on-the-fly strtof processing.
  // No overflow check here due to performance reason.
  const float log2_10 = std::log2(10.0f);
  if (*pos < input_size) {
    float fsign = 1.0f;
    float ftail = 0.1f;
    bool gotvalue = false;
    if (input[*pos] == '-') {
      fsign = -1.0f;
      ++(*pos);
    } else if (input[*pos] == '+') {
      ++(*pos);
    }
    *value = 0.0f;
    if (input[*pos] >= '0' && input[*pos] <= '9') {
      gotvalue = true;
    }
    while (*pos < input_size && input[*pos] >= '0' && input[*pos] <= '9') {
      (*value) = (*value) * 10.0f + static_cast<float>(input[*pos] - '0');
      ++(*pos);
    } // integer part
    if (*pos < input_size && input[*pos] == '.') {
      ++(*pos);
      if (*pos < input_size && input[*pos] >= '0' && input[*pos] <= '9') {
        gotvalue = true;
      }
      while (*pos < input_size && input[*pos] >= '0' && input[*pos] <= '9') {
        (*value) += ftail * static_cast<float>(input[*pos] - '0');
        ftail *= 0.1f;
        ++(*pos);
      }
    } // fraction part
    if (gotvalue) {
      (*value) *= fsign;
    } else {
      return false;
    }
    if (*pos < input_size && (input[*pos] == 'E' || input[*pos] == 'e')) {
      ++(*pos);
      if (*pos < input_size) {
        int64 fexp_int = 0;
        if (!GetValueOnTheFly(input, input_size, pos, &fexp_int)) {
          return false;
        }
        (*value) *= std::exp2(static_cast<float>(fexp_int) * log2_10);
        return true;
      }
    } else {
      return true;
    } // exponent part
  }
  return false;
}

template <typename T>
bool SplitNum(StringPiece &record, const char delim, std::vector<T>& result) {
  const char *input = record.data();
  size_t input_size = record.size();
  if (input_size && delim != '\0') {
    // No more than this number.
    result.reserve(input_size / 2 + 1);

    size_t i = 0;
    while(i < input_size) {
      while (i < input_size && input[i] == ' ') {
        ++i;
      }
      if (i < input_size) {
        T value = static_cast<T>(0);
        if (!GetValueOnTheFly(input, input_size, &i, &value)) {
          return false;
        }
        result.push_back(value);
        while (i < input_size && input[i] == ' ' && delim != ' ') {
          ++i;
        }
        if (i == input_size || input[i] == delim) {
          ++i;
        } else {
          return false;
        }
      } else if (delim == ' ') {
        // If delim is ' ', it is OK for redundant spaces at the end of line.
        return true;
      } else {
        // If delim is not ' ', only spaces found after a delim is irregular.
        return false;
      }
    }
  }
  return true;
}

template <typename T>
bool SplitKv(StringPiece &record, const char delim,
             std::vector<std::pair<int64, T> >& result) {
  const char *input = record.data();
  size_t input_size = record.size();

  if (input_size && delim != '\0') {
    // No more than this amount of kv pairs.
    result.reserve(input_size / 4 + 1);

    std::pair<int64, T> kv(0, static_cast<T>(0));
    int64& key = kv.first;
    T& value = kv.second;

    size_t i = 0;
    while(i < input_size) {
      // parsing key phase
      key = 0;
      while (i < input_size && input[i] == ' ') {
        ++i;
      }
      if (i < input_size) {
        if (!GetValueOnTheFly(input, input_size, &i, &key)) {
          return false;
        }
        while (i < input_size && input[i] == ' ' && delim != ' ') {
          ++i;
        }
        if (input[i] == ':') {
          ++i;
        } else {
          return false;
        }
      } else if (delim == ' ') {
        // If delim is ' ', it is OK for redundant spaces at the end of line.
        return true;
      } else {
        // If delim is not ' ', only spaces found after a delim is irregular.
        return false;
      } // key parsed, otherwise false returned.

      // parsing value phase.
      while (i < input_size && input[i] == ' ' && delim != ' ') {
        ++i;
      }
      value = static_cast<T>(0);
      if (!GetValueOnTheFly(input, input_size, &i, &value)) {
        return false;
      }
      result.push_back(kv);

      while (i < input_size && input[i] == ' ' && delim != ' ') {
        ++i;
      }
      if (i == input_size || input[i] == delim) {
        ++i;
      } else {
        return false;
      } // value parsed.
    }
  }
  return true;
}

} // namespace

class TransCsvID2SparseOp : public OpKernel {
 public:
  explicit TransCsvID2SparseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string delim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES(ctx, dtype_ == DT_INT32 || dtype_ == DT_INT64 || dtype_ == DT_FLOAT,
        errors::InvalidArgument("output data type ", dtype_, " not supported."));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_delim", &delim));
    OP_REQUIRES(ctx, (delim.size() == 1 && DelimSupported(delim[0])),
        errors::InvalidArgument("field_delim '", delim, "' not supported."));
    delim_ = delim[0];

    OP_REQUIRES_OK(ctx, ctx->GetAttr("id_as_value", &id_as_value_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* records;
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));

    const Tensor* max_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, (max_id >= 0 || max_id == ID_AUTO_DETECT_TAG),
                errors::InvalidArgument("invalid max_id setting: ", max_id));

    const Tensor* def_value;
    OP_REQUIRES_OK(ctx, ctx->input("default_value", &def_value));

    auto records_t = records->flat<string>();
    const int64 batch_size = records_t.size();

    std::vector<std::vector<int64>> cols_vec_(batch_size);

    mutex mu;
    size_t max_col_id GUARDED_BY(mu) = 0;
    Status status GUARDED_BY(mu);

    // Scan all the indicis and check the maximum column index in the matrix.
    auto doScan = [this, &status, &mu, ctx, &records_t, &cols_vec_, &max_col_id]
        (int64 start_i, int64 limit_i)
    {
      for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
        StringPiece record(records_t(batch_id));

        OP_CHECK_STATUS(status, mu,
            SplitNum<int64>(record, delim_, cols_vec_[batch_id]) == true,
            errors::InvalidArgument("Index record ", batch_id,
                                    " is not valid : ", records_t(batch_id)));
        int64 cols_len = cols_vec_[batch_id].size();
        if (cols_len > 0) {
          std::sort(cols_vec_[batch_id].begin(), cols_vec_[batch_id].end());
          if (cols_vec_[batch_id][cols_len-1] >= max_col_id) {
            mutex_lock l(mu);
            if (cols_vec_[batch_id][cols_len-1] >= max_col_id) {
              max_col_id = cols_vec_[batch_id][cols_len-1] + 1;
            }
          }
        }
      }
    };

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doScan);
    OP_REQUIRES_OK(ctx, status);

    if (max_id == ID_AUTO_DETECT_TAG) {
      max_id = max_col_id;
    } else {
      OP_REQUIRES(ctx, max_id >= max_col_id,
                  errors::InvalidArgument("max_id set as ", max_id,
                      " but less then real maximum col id ", max_col_id));
    }

    int64 offset_[batch_size + 1];
    offset_[0] = 0;
    for (int64 batch_id = 0; batch_id < batch_size; ++batch_id) {
      offset_[batch_id + 1] = offset_[batch_id] + cols_vec_[batch_id].size();
    }

    Tensor indices(DT_INT64, TensorShape({offset_[batch_size], 2}));
    Tensor values(dtype_, TensorShape({offset_[batch_size]}));
    Tensor dense_shape(DT_INT64, TensorShape({2}));

    auto doFill = [this, &status, &mu, &values, &indices, &offset_, &def_value, &cols_vec_]
        (int64 start_i, int64 limit_i)
    {
      switch (dtype_) {
        case DT_INT32:
          Fill<int32>(values, indices, offset_, def_value, cols_vec_, start_i, limit_i);
          break;
        case DT_INT64:
          Fill<int64>(values, indices, offset_, def_value, cols_vec_, start_i, limit_i);
          break;
        case DT_FLOAT:
          Fill<float>(values, indices, offset_, def_value, cols_vec_, start_i, limit_i);
          break;
        default:
          OP_CHECK_STATUS(status, mu, false,
              errors::InvalidArgument("output data type ", dtype_, " not supported."));
      }
    };

    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doFill);
    OP_REQUIRES_OK(ctx, status);

    auto dense_shape_ = dense_shape.flat<int64>();
    dense_shape_(0) = batch_size;
    dense_shape_(1) = max_id;
    ctx->set_output(0, indices);
    ctx->set_output(1, values);
    ctx->set_output(2, dense_shape);
  }

 private:
  DataType dtype_;
  char delim_;
  bool id_as_value_;

  template <typename T>
  void Fill(Tensor& values, Tensor& indices,
            const int64* offset_, const Tensor* def_value,
            const std::vector<std::vector<int64>>& cols_vec_,
            const int64& start_i, const int64& limit_i) {
    const T def_value_ = def_value->scalar<T>()();
    auto values_ = values.vec<T>();
    auto indices_ = indices.matrix<int64>();
    for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
      const int64 cols_len = cols_vec_[batch_id].size();
      const int64 offset = offset_[batch_id];

      for (int64 index = 0; index < cols_len; ++index) {
        indices_(index + offset, 0) = batch_id;
        indices_(index + offset, 1) = cols_vec_[batch_id][index];
        if (this->id_as_value_) {
          values_(index + offset) = static_cast<T>(cols_vec_[batch_id][index]);
        } else {
          values_(index + offset) = def_value_;
        }
      }
    }
  };
};

class TransCsvID2DenseOp : public OpKernel {
 public:
  explicit TransCsvID2DenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string delim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES(ctx, dtype_ == DT_INT32 || dtype_ == DT_INT64 || dtype_ == DT_FLOAT,
        errors::InvalidArgument("output data type ", dtype_, " not supported."));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_delim", &delim));
    OP_REQUIRES(ctx, (delim.size() == 1 && DelimSupported(delim[0])),
        errors::InvalidArgument("field_delim '", delim, "' not supported."));
    delim_ = delim[0];

    OP_REQUIRES_OK(ctx, ctx->GetAttr("id_as_value", &id_as_value_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* records;
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));

    const Tensor* max_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, (max_id >= 0 || max_id == ID_AUTO_DETECT_TAG),
                errors::InvalidArgument("invalid max_id setting: ", max_id));

    const Tensor* def_value;
    OP_REQUIRES_OK(ctx, ctx->input("default_value", &def_value));

    auto records_t = records->flat<string>();
    const int64 batch_size = records_t.size();

    std::vector<std::vector<int64>> cols_vec_(batch_size);

    mutex mu;
    size_t max_col_id GUARDED_BY(mu) = 0;
    Status status GUARDED_BY(mu);

    // Scan all the indicis and check the maximum column index in the matrix.
    auto doScan = [this, &status, &mu, ctx, &records_t, &cols_vec_, &max_col_id]
        (int64 start_i, int64 limit_i)
    {
      for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
        StringPiece record(records_t(batch_id));

        OP_CHECK_STATUS(status, mu,
            SplitNum<int64>(record, delim_, cols_vec_[batch_id]) == true,
            errors::InvalidArgument("Index record ", batch_id,
                                    " is not valid : ", records_t(batch_id)));
        int64 cols_len = cols_vec_[batch_id].size();
        if (cols_len > 0) {
          std::sort(cols_vec_[batch_id].begin(), cols_vec_[batch_id].end());
          if (cols_vec_[batch_id][cols_len-1] >= max_col_id) {
            mutex_lock l(mu);
            if (cols_vec_[batch_id][cols_len-1] >= max_col_id) {
              max_col_id = cols_vec_[batch_id][cols_len-1] + 1;
            }
          }
        }
      }
    };

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doScan);
    OP_REQUIRES_OK(ctx, status);

    if (max_id == ID_AUTO_DETECT_TAG) {
      max_id = max_col_id;
    } else {
      OP_REQUIRES(ctx, max_id >= max_col_id,
                  errors::InvalidArgument("max_id set as ", max_id,
                      " but less then real maximum col id ", max_col_id));
    }

    Tensor values(dtype_, TensorShape({batch_size, max_id}));

    auto doFill = [this, &status, &mu, &values, &max_id, &def_value, &cols_vec_]
        (int64 start_i, int64 limit_i)
    {
      switch (dtype_) {
        case DT_INT32:
          Fill<int32>(values, max_id, def_value, cols_vec_, start_i, limit_i);
          break;
        case DT_INT64:
          Fill<int64>(values, max_id, def_value, cols_vec_, start_i, limit_i);
          break;
        case DT_FLOAT:
          Fill<float>(values, max_id, def_value, cols_vec_, start_i, limit_i);
          break;
        default:
          OP_CHECK_STATUS(status, mu, false,
              errors::InvalidArgument("output data type ", dtype_, " not supported."));
      }
    };

    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doFill);
    OP_REQUIRES_OK(ctx, status);

    ctx->set_output(0, values);
 }

 private:
  DataType dtype_;
  char delim_;
  bool id_as_value_;

  template <typename T>
  void Fill(Tensor& values, const int64& max_id, const Tensor* def_value,
            const std::vector<std::vector<int64>>& cols_vec_,
            const int64& start_i, const int64& limit_i) {
    const T def_value_ = def_value->scalar<T>()();
    auto values_ = values.matrix<T>();

    for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
      for (int64 index = 0; index < max_id; ++index) {
        values_(batch_id, index) = static_cast<T>(0);
      }

      int64 cols_len = cols_vec_[batch_id].size();
      for (int64 i = 0; i < cols_len; ++i) {
        int64 index = cols_vec_[batch_id][i];
        if (this->id_as_value_) {
          values_(batch_id, index) = static_cast<T>(index);
        } else {
          values_(batch_id, index) = def_value_;
        }
      }
    }
  };
};

class TransCsvKV2SparseOp : public OpKernel {
 public:
  explicit TransCsvKV2SparseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string delim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES(ctx, dtype_ == DT_INT32 || dtype_ == DT_INT64 || dtype_ == DT_FLOAT,
        errors::InvalidArgument("output data type ", dtype_, " not supported."));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_delim", &delim));
    OP_REQUIRES(ctx, (delim.size() == 1 && delim[0] != ':' && DelimSupported(delim[0])),
        errors::InvalidArgument("field_delim '", delim, "' not supported."));
    delim_ = delim[0];
  }

  void Compute(OpKernelContext* ctx) override {
    switch (dtype_) {
      case DT_INT32:
        ComputeInternal<int32>(ctx);
        break;
      case DT_INT64:
        ComputeInternal<int64>(ctx);
        break;
      case DT_FLOAT:
        ComputeInternal<float>(ctx);
        break;
      default:
        ctx->SetStatus(errors::InvalidArgument(
            "output data type ", dtype_, " not supported."));
    }
  }

 private:
  DataType dtype_;
  char delim_;

 private:
  template <typename T>
  void ComputeInternal(OpKernelContext* ctx) {
    const Tensor* records;
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));

    const Tensor* max_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, (max_id >= 0 || max_id == ID_AUTO_DETECT_TAG),
                errors::InvalidArgument("invalid max_id setting: ", max_id));

    auto records_t = records->flat<string>();
    const int64 batch_size = records_t.size();

    std::vector<std::vector<std::pair<int64, T>>> cols_vec_(batch_size);

    mutex mu;
    size_t max_col_id GUARDED_BY(mu) = 0;
    Status status GUARDED_BY(mu);

    // Scan and store k/v pairs, check the maximum column index in the matrix.
    auto doScan = [this, &status, &mu, ctx, &records_t, &cols_vec_, &max_col_id]
        (int64 start_i, int64 limit_i)
    {
      for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
        StringPiece record(records_t(batch_id));

        OP_CHECK_STATUS(status, mu,
                        SplitKv(record, delim_, cols_vec_[batch_id]) == true,
                        errors::InvalidArgument("kv in record ", batch_id,
                            " is not valid : ", records_t(batch_id)));
        int64 cols_len = cols_vec_[batch_id].size();
        if (cols_len > 0) {
          std::sort(cols_vec_[batch_id].begin(), cols_vec_[batch_id].end());
          if (cols_vec_[batch_id][cols_len-1].first >= max_col_id) {
            mutex_lock l(mu);
            if (cols_vec_[batch_id][cols_len-1].first >= max_col_id) {
              max_col_id = cols_vec_[batch_id][cols_len-1].first + 1;
            }
          }
        }
      }
    };

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doScan);
    OP_REQUIRES_OK(ctx, status);

    if (max_id == ID_AUTO_DETECT_TAG) {
      max_id = max_col_id;
    } else {
      OP_REQUIRES(ctx, max_id >= max_col_id,
                  errors::InvalidArgument("max_id set as ", max_id,
                      " but less then real maximum col id ", max_col_id));
    }

    int64 offset_[batch_size + 1];
    offset_[0] = 0;
    for (int64 batch_id = 0; batch_id < batch_size; ++batch_id) {
      offset_[batch_id + 1] = offset_[batch_id] + cols_vec_[batch_id].size();
    }

    Tensor indices(DT_INT64, TensorShape({offset_[batch_size], 2}));
    Tensor values(dtype_, TensorShape({offset_[batch_size]}));
    Tensor dense_shape(DT_INT64, TensorShape({2}));

    auto doFill = [this, &values, &indices, &offset_, &cols_vec_]
        (int64 start_i, int64 limit_i)
    {
      Fill(values, indices, offset_, cols_vec_, start_i, limit_i);
    };

    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doFill);
    OP_REQUIRES_OK(ctx, status);

    auto dense_shape_ = dense_shape.flat<int64>();
    dense_shape_(0) = batch_size;
    dense_shape_(1) = max_id;
    ctx->set_output(0, indices);
    ctx->set_output(1, values);
    ctx->set_output(2, dense_shape);
  }

  template <typename T>
  void Fill(Tensor& values, Tensor& indices, const int64* offset_,
            const std::vector<std::vector<std::pair<int64, T>>>& cols_vec_,
            const int64& start_i, const int64& limit_i) {
    auto values_ = values.vec<T>();
    auto indices_ = indices.matrix<int64>();
    for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
      const int64 cols_len = cols_vec_[batch_id].size();
      const int64 offset = offset_[batch_id];

      for (int64 index = 0; index < cols_len; ++index) {
        indices_(index + offset, 0) = batch_id;
        indices_(index + offset, 1) = cols_vec_[batch_id][index].first;
        values_(index + offset) = cols_vec_[batch_id][index].second;
      }
    }
  };
};

class TransCsvKV2DenseOp : public OpKernel {
 public:
  explicit TransCsvKV2DenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string delim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES(ctx, dtype_ == DT_INT32 || dtype_ == DT_INT64 || dtype_ == DT_FLOAT,
        errors::InvalidArgument("output data type ", dtype_, " not supported."));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_delim", &delim));
    OP_REQUIRES(ctx, (delim.size() == 1 && delim[0] != ':' && DelimSupported(delim[0])),
        errors::InvalidArgument("field_delim is not one char or not supported."));
    delim_ = delim[0];
  }

  void Compute(OpKernelContext* ctx) override {
    switch (dtype_) {
      case DT_INT32:
        ComputeInternal<int32>(ctx);
        break;
      case DT_INT64:
        ComputeInternal<int64>(ctx);
        break;
      case DT_FLOAT:
        ComputeInternal<float>(ctx);
        break;
      default:
        ctx->SetStatus(errors::InvalidArgument(
            "output data type ", dtype_, " not supported."));
    }
  }

 private:
  DataType dtype_;
  char delim_;

 private:
  template <typename T>
  void ComputeInternal(OpKernelContext* ctx) {
    const Tensor* records;
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));

    const Tensor* max_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, (max_id >= 0 || max_id == ID_AUTO_DETECT_TAG),
                errors::InvalidArgument("invalid max_id setting: ", max_id));

    auto records_t = records->flat<string>();
    const int64 batch_size = records_t.size();

    std::vector<std::vector<std::pair<int64, T>>> cols_vec_(batch_size);

    mutex mu;
    size_t max_col_id GUARDED_BY(mu) = 0;
    Status status GUARDED_BY(mu);

    // Scan and store k/v pairs, check the maximum column index in the matrix.
    auto doScan = [this, &status, &mu, ctx, &records_t, &cols_vec_, &max_col_id]
        (int64 start_i, int64 limit_i)
    {
      for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
        StringPiece record(records_t(batch_id));

        OP_CHECK_STATUS(status, mu,
                        SplitKv(record, delim_, cols_vec_[batch_id]) == true,
                        errors::InvalidArgument("kv in record ", batch_id,
                            " is not valid : ", records_t(batch_id)));
        int64 cols_len = cols_vec_[batch_id].size();
        if (cols_len > 0) {
          std::sort(cols_vec_[batch_id].begin(), cols_vec_[batch_id].end());
          if (cols_vec_[batch_id][cols_len-1].first >= max_col_id) {
            mutex_lock l(mu);
            if (cols_vec_[batch_id][cols_len-1].first >= max_col_id) {
              max_col_id = cols_vec_[batch_id][cols_len-1].first + 1;
            }
          }
        }
      }
    };

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doScan);
    OP_REQUIRES_OK(ctx, status);

    if (max_id == ID_AUTO_DETECT_TAG) {
      max_id = max_col_id;
    } else {
      OP_REQUIRES(ctx, max_id >= max_col_id,
                  errors::InvalidArgument("max_id set as ", max_id,
                      " but less then real maximum col id ", max_col_id));
    }

    Tensor values(dtype_, TensorShape({batch_size, max_id}));

    auto doFill = [this, &values, &max_id, &cols_vec_]
        (int64 start_i, int64 limit_i)
    {
      Fill(values, max_id, cols_vec_, start_i, limit_i);
    };

    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doFill);
    OP_REQUIRES_OK(ctx, status);

    ctx->set_output(0, values);
  }

  template <typename T>
  void Fill(Tensor& values, const int64& max_id,
            const std::vector<std::vector<std::pair<int64, T>>>& cols_vec_,
            const int64& start_i, const int64& limit_i) {
    auto values_ = values.matrix<T>();

    for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
      for (int64 index = 0; index < max_id; ++index) {
        values_(batch_id, index) = static_cast<T>(0);
      }

      int64 cols_len = cols_vec_[batch_id].size();
      for (int64 i = 0; i < cols_len; ++i) {
        int64 index = cols_vec_[batch_id][i].first;
        T value = cols_vec_[batch_id][i].second;
        values_(batch_id, index) = value;
      }
    }
  };
};

class TransCsvToDenseOp : public OpKernel {
 public:
  explicit TransCsvToDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string delim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES(ctx, dtype_ == DT_INT32 || dtype_ == DT_INT64 || dtype_ == DT_FLOAT,
        errors::InvalidArgument("output data type ", dtype_, " not supported."));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_delim", &delim));
    OP_REQUIRES(ctx, (delim.size() == 1 && DelimSupported(delim[0])),
        errors::InvalidArgument("field_delim is not one char or not supported."));
    delim_ = delim[0];
  }

  void Compute(OpKernelContext* ctx) override {
    switch (dtype_) {
      case DT_INT32:
        ComputeInternal<int32>(ctx);
        break;
      case DT_INT64:
        ComputeInternal<int64>(ctx);
        break;
      case DT_FLOAT:
        ComputeInternal<float>(ctx);
        break;
      default:
        ctx->SetStatus(errors::InvalidArgument(
            "output data type ", dtype_, " not supported."));
    }
  }

 private:
  DataType dtype_;
  char delim_;

 private:
  template <typename T>
  void ComputeInternal(OpKernelContext* ctx) {
    const Tensor* records;
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));

    const Tensor* max_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, (max_id >= 0 || max_id == ID_AUTO_DETECT_TAG),
                errors::InvalidArgument("invalid max_id setting: ", max_id));

    auto records_t = records->flat<string>();
    const int64 batch_size = records_t.size();

    std::vector<std::vector<T>> cols_vec_(batch_size);

    mutex mu;
    size_t max_col_id GUARDED_BY(mu) = 0;
    Status status GUARDED_BY(mu);

    // Scan and store values, check the maximum column index in the matrix.
    auto doScan = [this, &status, &mu, ctx, &records_t, &cols_vec_, &max_col_id]
        (int64 start_i, int64 limit_i)
    {
      for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
        StringPiece record(records_t(batch_id));

        OP_CHECK_STATUS(status, mu,
                        SplitNum(record, delim_, cols_vec_[batch_id]) == true,
                        errors::InvalidArgument("values in record ", batch_id,
                            " is not valid : ", records_t(batch_id)));
        int64 cols_len = cols_vec_[batch_id].size();
        if (cols_len > max_col_id) {
          mutex_lock l(mu);
          if (cols_len > max_col_id) {
            max_col_id = cols_len;
          }
        }
      }
    };

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doScan);
    OP_REQUIRES_OK(ctx, status);

    if (max_id == ID_AUTO_DETECT_TAG) {
      max_id = max_col_id;
    } else {
      OP_REQUIRES(ctx, max_id >= max_col_id,
                  errors::InvalidArgument("max_id set as ", max_id,
                      " but less then real maximum col id ", max_col_id));
    }

    Tensor values(dtype_, TensorShape({batch_size, max_id}));

    auto doFill = [this, &values, &max_id, &cols_vec_]
        (int64 start_i, int64 limit_i)
    {
      Fill(values, max_id, cols_vec_, start_i, limit_i);
    };

    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doFill);
    OP_REQUIRES_OK(ctx, status);

    ctx->set_output(0, values);
  }

  template <typename T>
  void Fill(Tensor& values, const int64& max_id,
            const std::vector<std::vector<T>>& cols_vec_,
            const int64& start_i, const int64& limit_i) {
    auto values_ = values.matrix<T>();

    for (int64 batch_id = start_i; batch_id < limit_i; ++batch_id) {
      int64 cols_len = cols_vec_[batch_id].size();
      for (int64 i = 0; i < cols_len; ++i) {
        T value = cols_vec_[batch_id][i];
        values_(batch_id, i) = value;
      }
      for (int64 i = cols_len; i < max_id; ++i) {
        values_(batch_id, i) = static_cast<T>(0);
      }
    }
  };
};


REGISTER_KERNEL_BUILDER(Name("TransCsvID2Sparse").Device(DEVICE_CPU), TransCsvID2SparseOp);
REGISTER_KERNEL_BUILDER(Name("TransCsvID2Dense").Device(DEVICE_CPU), TransCsvID2DenseOp);
REGISTER_KERNEL_BUILDER(Name("TransCsvKV2Sparse").Device(DEVICE_CPU), TransCsvKV2SparseOp);
REGISTER_KERNEL_BUILDER(Name("TransCsvKV2Dense").Device(DEVICE_CPU), TransCsvKV2DenseOp);
REGISTER_KERNEL_BUILDER(Name("TransCsvToDense").Device(DEVICE_CPU), TransCsvToDenseOp);

}  // namespace tensorflow

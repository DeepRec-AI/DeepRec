/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/string_to_hash_bucket_ali_op.h"

#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/strong_hash.h"

namespace tensorflow {

// StringToHashBucket is deprecated in favor of StringToHashBucketFast/Strong.
REGISTER_KERNEL_BUILDER(Name("StringToHashBucketFast").Device(DEVICE_CPU),
                        StringToHashBucketBatchAliOp<Fingerprint64>);

REGISTER_KERNEL_BUILDER(Name("StringToHashBucketStrong").Device(DEVICE_CPU),
                        StringToKeyedHashBucketAliOp<StrongKeyedHash>);

REGISTER_KERNEL_BUILDER(Name("StringToHash64").Device(DEVICE_CPU),
                        StringToHash64Op<Fingerprint64>);

}  // namespace tensorflow

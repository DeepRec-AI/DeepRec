/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.cuh"

namespace SparseOperationKit {

// TODO: optimize this function
__global__ void reduce_sum(const size_t* nums, const size_t nums_len, size_t* result) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 == gid && result != nullptr) {
        result[0] = 0;
        for (size_t j = 0; j < nums_len; j++)
            result[0] += nums[j];
    }
}

} // namespace SparseOperationKit
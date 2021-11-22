"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys
sys.path.append("../../../")
import sparse_operation_kit
sys.path.append("../../../unit_test/test_scripts/")
from utils import *

def TFDataset(filename, batchsize, as_sparse_tensor, repeat=1):
    samples, labels = restore_from_file(filename)
    dataset = tf_dataset(keys=samples, labels=labels,
                         batchsize=batchsize,
                         to_sparse_tensor=as_sparse_tensor,
                         repeat=repeat)
    del samples
    del labels
    return dataset


def test_DALI():
    from nvidia.dali import pipeline_def, Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    import nvidia.dali.plugin.tf as dali_tf
    import numpy as np
    
    @pipeline_def(device_id=0)
    def data_pipeline():
        # samples, labels = restore_from_file(r"./data.file")
        samples = np.ones(shape=(100, 10), dtype=np.int64)
        labels = np.ones(shape=(1,), dtype=np.float32)
        return types.Constant(value=samples, dtype=types.DALIDataType.INT64, device='cpu'),\
               types.Constant(value=labels, dtype=types.DALIDataType.FLOAT, device='cpu')

    pipeline = data_pipeline()
    
    dataset = dali_tf.DALIDataset(pipeline=pipeline,
                                  batch_size=2,
                                  output_shapes=((2, 100, 10), (2, 1)),
                                  output_dtypes=(tf.int64, tf.int32),
                                  device_id=0)

    for i, (keys, labels) in enumerate(dataset):
        print("Iter: {}, keys: {}, labels: {}".format(i, keys, labels))
        break

if __name__ == "__main__":
    # dataset = TFDataset(filename=r"./datas.file",
    #                     batchsize=2,
    #                     as_sparse_tensor=False)

    # for i, (keys, labels) in enumerate(dataset):
    #     print("iteration: {}, keys={}, labels={}".format(i, keys.shape, labels.shape))
    #     break
    test_DALI()
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functional tests for async embedding."""

from tensorflow.python.platform import test
from tensorflow.python.training import async_embedding_stage
from tensorflow.python.feature_column import feature_column
from tensorflow.python.ops import prefetch
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

class AsyncEmbeddingStageTest(test.TestCase):
    def testAsyncEmbedding(self):
        dataset = dataset_ops.Dataset.from_tensor_slices(({'a': [1, 2, 3]}))
        dataset = dataset.batch(3)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        next_element = prefetch.staged(next_element)

        features = next_element
        a = feature_column._categorical_column_with_identity('a', num_buckets=5, default_value=0)
        a_one_hot = feature_column._indicator_column(a)
        a_one_hot_dense = feature_column.input_layer(features, a_one_hot)

        b = variables.Variable(array_ops.ones([5, 10]))
        c = math_ops.matmul(a_one_hot_dense, b)

        graph = ops.get_default_graph()

        stage_put_num = 0
        stage_take_num = 0
        for node in graph.get_operations():
            if node.type == 'TensorBufferPut':
                stage_put_num += 1
            elif node.type == 'TensorBufferTake':
                stage_take_num += 1

        self.assertEqual(stage_put_num, 1)
        self.assertEqual(stage_take_num, 1)

        ae = async_embedding_stage.AsyncEmbeddingStage(1, 1, None)
        ae.stage(graph)

        stage_put_num = 0
        stage_take_num = 0
        for node in graph.get_operations():
            if node.type == 'TensorBufferPut':
                stage_put_num += 1
            elif node.type == 'TensorBufferTake':
                stage_take_num += 1

        self.assertEqual(stage_put_num, 2)
        self.assertEqual(stage_take_num, 2)
        
        
if __name__ == "__main__":
    test.main()

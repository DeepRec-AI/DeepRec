# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for string_split_and_pad_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import googletest


class StringSplitAndPadTest(test_util.TensorFlowTestCase):

  def testSplitWithoutPad(self):
    x_1 = "f1,f2,f3,f4"
    with self.test_session(use_gpu=True):
      y_tf = string_ops.string_split_and_pad(
          [x_1], max_length=4, delimiter=",").eval()
      expect = [[b'f1', b'f2', b'f3', b'f4']]
      re = y_tf == expect
      for line in re:
        for item in line:
          self.assertEqual(item, True)

    x_2 = "f5,f6,f7,f8"
    with self.test_session(use_gpu=True):
      y_tf = string_ops.string_split_and_pad(
          [x_1, x_2], max_length=4, delimiter=",").eval()
      expect = [[b'f1', b'f2', b'f3', b'f4'], [b'f5', b'f6', b'f7', b'f8']]
      re = y_tf == expect
      for line in re:
        for item in line:
          self.assertEqual(item, True)

  def testSplitWithPad(self):
    x_1 = "f1,f2,f3,f4"
    with self.test_session(use_gpu=True):
      y_tf = string_ops.string_split_and_pad(
          [x_1], max_length=5, delimiter=",").eval()
      expect = [[b'f1', b'f2', b'f3', b'f4', b'</s>']]
      re = y_tf == expect
      for line in re:
        for item in line:
          self.assertEqual(item, True)

    x_2 = "f5,f6,f7,f8"
    with self.test_session(use_gpu=True):
      y_tf = string_ops.string_split_and_pad(
          [x_1, x_2], max_length=6, delimiter=",").eval()
      expect = [[b'f1', b'f2', b'f3', b'f4', b'</s>', b'</s>'],
                [b'f5', b'f6', b'f7', b'f8', b'</s>', b'</s>']]
      re = y_tf == expect
      for line in re:
        for item in line:
          self.assertEqual(item, True)

  def testInputInvalidAxis(self):
    x = "f1,f2,f3,f4"
    with self.assertRaisesRegexp(
        ValueError, "Shape must be rank 1 but is rank 0"):
      string_ops.string_split_and_pad(x, max_length=4, delimiter=",")


if __name__ == "__main__":
  googletest.main()

"""Tests for Structured Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.structured_model.python.core import *
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class StructuredModelTest(test.TestCase):

  def testFindBounderyTensors(self):
    a = constant_op.constant(1, name='a')
    b = constant_op.constant(2, name='b')
    c = constant_op.constant(3, name='c')

    d = (a + b) * c
    user_op_sets, item_op_sets, boundery_tensor_sets = find_boundery_tensors(
      user_ops=[a.op, b.op], item_ops=[c.op])

    self.assertEqual("add", list(user_op_sets)[0].name)
    self.assertEqual("mul", list(item_op_sets)[0].name)
    self.assertEqual("add:0", list(boundery_tensor_sets)[0].name)


if __name__ == "__main__":
  test.main()

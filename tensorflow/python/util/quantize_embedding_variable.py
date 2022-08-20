# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Exposes the Python wrapper for quantize embedding variable."""
from __future__ import absolute_import, division, print_function

from tensorflow.python.pywrap_tensorflow import QuantizeEmbeddingVariablesByName
from tensorflow.python.util import compat


def quantize_by_name(
    input_prefix, output_prefix, names, quant_names, scale_names, dtype
):
  """Python wrapper for quantize embedding variable.

  Args:
    input_prefix: String. Prefix of input checkpoint.
    output_prefix: String. Prefix of output checkpoint.
    names: List of tensor names to be quantized.
    quant_names: List of quantized tensor names.
    scale_names: List of scale tensor names.
    dtype: tf.bfloat16 or tf.int8
  """
  input_prefix = compat.as_bytes(input_prefix)
  output_prefix = compat.as_bytes(output_prefix)
  names_string = compat.as_bytes(",".join(names))
  quant_names_string = compat.as_bytes(",".join(quant_names))
  scale_names_string = compat.as_bytes(",".join(scale_names))
  QuantizeEmbeddingVariablesByName(
      input_prefix,
      output_prefix,
      names_string,
      quant_names_string,
      scale_names_string,
      dtype.as_datatype_enum,
  )

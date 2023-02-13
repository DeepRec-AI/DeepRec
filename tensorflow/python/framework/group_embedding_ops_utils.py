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
"""Defines functions common to group embedding lookup files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum, unique

@unique
class FUSIONTYPE(Enum):
  COLLECTIVE = "collective"
  DISTRIBUTED = "ps"
  LOCALIZED = "localized"
  UNKNOWN = "unknown"

_group_lookup_fusion_type = FUSIONTYPE.UNKNOWN

def set_group_lookup_fusion_type(fusion_type):
  def str_to_fusion_type(fusion_type):
    if fusion_type == "collective":
      return FUSIONTYPE.COLLECTIVE
    elif fusion_type == "ps":
      return FUSIONTYPE.DISTRIBUTED
    elif fusion_type == "localized":
      return FUSIONTYPE.LOCALIZED

  global _group_lookup_fusion_type
  _group_lookup_fusion_type = str_to_fusion_type(fusion_type)

def get_group_lookup_fusion_type():
  global _group_lookup_fusion_type
  return _group_lookup_fusion_type
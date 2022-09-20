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

%include <std_string.i>
%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/c/quantize_embedding_variable.h"
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::checkpoint;
%unignore QuantizeEmbeddingVariablesByName;
%unignore RemoveVariablesByName;

%{
void QuantizeEmbeddingVariablesByName(string input_prefix, string output_prefix,
                                      string names_string,
                                      string quant_names_string,
                                      string scale_names_string,
                                      TF_DataType data_type, bool is_ev) {
  std::vector<string> names = tensorflow::str_util::Split(names_string, ',');
  std::vector<string> quant_names =
      tensorflow::str_util::Split(quant_names_string, ',');
  std::vector<string> scale_names =
      tensorflow::str_util::Split(scale_names_string, ',');

  tensorflow::checkpoint::QuantizeEmbeddingVariable(
      input_prefix, output_prefix, names, quant_names, scale_names, data_type,
      is_ev);
}
%}

void QuantizeEmbeddingVariablesByName(string input_prefix, string output_prefix,
                                      string names_string,
                                      string quant_names_string,
                                      string scale_names_string,
                                      TF_DataType data_type, bool is_ev);

%{
void RemoveVariablesByName(string input_prefix, string output_prefix,
                           string names_string) {
  std::vector<string> names = tensorflow::str_util::Split(names_string, ',');
  tensorflow::checkpoint::RemoveVariable(input_prefix, output_prefix, names);
}
%}

void RemoveVariablesByName(string input_prefix, string output_prefix,
                           string names_string);

%unignoreall

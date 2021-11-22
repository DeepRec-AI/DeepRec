#!/bin/bash
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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
# =============================================================================

die() {
  echo $@
  exit 1
}

to_lower () {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

to_upper () {
  echo "$1" | tr '[:lower:]' '[:upper:]'
}

# Download tensorflow whl package
function prepare_env() {
  yes | pip3 install $1
}

# prepare to test
function model_test() {
    model_name=$(to_upper "$1")
    pushd modelzoo/$model_name
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo -e "---> start test \033[32m$model_name\033[0m..."
    python train.py $train_options --data_location=$data_location/$model_name
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    popd
}

function main_execute(){
    _test_model=$1
    case $_test_model in
    "ALL")
        for single_model in "DIEN" "DIN" "DLRM" "DSSM" "WDL"
        do
            main_execute $single_model
        done
        ;;
    "DIEN")
        model_test $_test_model
        ;;

    "DIN")
        model_test $_test_model
        ;;

    "DLRM")
        model_test $_test_model
        ;;

    "DSSM")
        model_test $_test_model
        ;;

    "WDL")
        model_test $_test_model
        ;;

    *)
        die "Unknown model type == $_test_model"
        ;;
    esac
}

test_model=$(to_upper "$1")
tf_whl=$2
data_location=$3
shift 3

train_options=$@

prepare_env $tf_whl
main_execute $test_model

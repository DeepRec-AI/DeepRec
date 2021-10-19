#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
#

# Script used to install SparseOperationKit to your system path.
#!/bin/bash
set -e

# judge whether it is executed by root user
user=`id -u`
if [ $user -ne 0 ]; then 
echo "[ERROR]: You don't have enough privilege to install SOK."
exit
fi

# get arguments from command line
Args=`getopt -o -h -l SM::,USE_NVTX:: -- "$@"`
HelpMessage='Install command: ./install.sh --SM=[GPU Compute Capability] --USE_NVTX=[ON/OFF]'
eval set -- "${Args}"
if [ $# -eq 1 ] && [ $1 = "--" ]; then
    echo "[INFO]: Using default options."
fi
while true ; do
    case "$1" in
        -h) echo $HelpMessage ; exit 0 ;;
        --SM) SM=$2 ; shift 2 ;;
        --USE_NVTX) 
            case "$2" in
                "ON") USE_NVTX="ON" ; shift 2 ;;
                "OFF") USE_NVTX="OFF" ; shift 2 ;;
                *) echo "[ERROR] Unrecognized option. $HelpMessage" ; exit 1 ;;
            esac ;;
        --) shift ;;
        *) break ;;
    esac
done

# compile SOK from source and install lib to system path
workdir=$(pwd)
mkdir -p build && cd build && rm -rf * && cmake -DSM=$SM -DUSE_NVTX=$USE_NVTX .. && make -j && make install
cd $workdir
cp -r $workdir/sparse_operation_kit/ /usr/local/lib/sparse_operation_kit

# set PYTHONPATH
echo "export PYTHONPATH=/usr/local/lib/:$PYTHONPATH" > /etc/profile
source /etc/profile
echo "export PYTHONPATH=/usr/local/lib/:$PYTHONPATH" > ~/.bashrc
source ~/.bashrc

echo "Successfully installed SparseOperationKit to /usr/local/lib/sparse_operation_kit."

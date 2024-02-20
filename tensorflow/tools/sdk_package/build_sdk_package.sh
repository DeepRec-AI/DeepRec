#!/usr/bin/env bash
# Copyright 2024 The DeepRec Authors. All Rights Reserved.
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

# This script is used for packaging TensorFlow SDK files into a tarball.
# The processing flow took 'tensorflow/tools/pip_package/build_pip_package.sh'
# as the reference.

set -e

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
function is_windows() {
  # On windows, the shell script is actually running in msys
  if [[ "${PLATFORM}" =~ msys_nt* ]]; then
    true
  else
    false
  fi
}

function main() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  DEST=$1
  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
  mkdir -p "${TMPDIR}/sdk/bin"
  mkdir -p "${TMPDIR}/sdk/include"
  mkdir -p "${TMPDIR}/sdk/lib"

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  if is_windows; then
    echo "Windows version TensorFlow SDK not supported..."
  elif [ ! -d bazel-bin/tensorflow/tools/sdk_package/build_sdk_package.runfiles/org_tensorflow ]; then
    # Really old (0.2.1-) runfiles, without workspace name.
    echo "TensorFlow SDK does not support such old verions..."
  else
    RUNFILES=bazel-bin/tensorflow/tools/sdk_package/build_sdk_package.runfiles/org_tensorflow
    if [ -d ${RUNFILES}/external ]; then
      # Old-style runfiles structure (--legacy_external_runfiles).
      cp -RL ${RUNFILES}/tensorflow "${TMPDIR}/sdk/include"
      # Check LLVM headers for XLA support.
      if [ -d ${RUNFILES}/external/llvm_archive ]; then
        # Old-style runfiles structure (--legacy_external_runfiles).
        mkdir -p ${TMPDIR}/sdk/include/external/llvm/include
        cp -RL ${RUNFILES}/external/llvm_archive/include/llvm \
          "${TMPDIR}/sdk/include/external/llvm/include"
        pushd ${TMPDIR}/sdk/include
        ln -s external/llvm/include/llvm llvm
        popd
      fi
      # Copy MKL libs over so they can be loaded at runtime
      so_lib_dir=$(ls $RUNFILES | grep solib) || true
      if [ -n "${so_lib_dir}" ]; then
        mkl_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep mkl) || true
        if [ -n "${mkl_so_dir}" ]; then
          cp -L ${RUNFILES}/${so_lib_dir}/${mkl_so_dir}/*.so "${TMPDIR}/sdk/lib"
        fi
      fi
    else
      # New-style runfiles structure (--nolegacy_external_runfiles).
      cp -RL ${RUNFILES}/tensorflow "${TMPDIR}/sdk/include"
      # Check LLVM headers for XLA support.
      if [ -d bazel-bin/tensorflow/tools/sdk_package/build_sdk_package.runfiles/llvm_archive ]; then
        cp -RL \
          bazel-bin/tensorflow/tools/sdk_package/build_sdk_package.runfiles/llvm_archive/include/llvm \
          "${TMPDIR}/sdk/include"
      fi
      # Copy MKL libs over so they can be loaded at runtime
      so_lib_dir=$(ls $RUNFILES | grep solib) || true
      if [ -n "${so_lib_dir}" ]; then
        mkl_so_dir=$(ls ${RUNFILES}/${so_lib_dir} | grep mkl) || true
        if [ -n "${mkl_so_dir}" ]; then
          cp -L ${RUNFILES}/${so_lib_dir}/${mkl_so_dir}/*.so "${TMPDIR}/sdk/lib"
        fi
      fi
    fi
  fi

  # move and strip the dynamic library file for packaging.
  # at default the .so file was not writable for the owner,
  # so using a 'chmod +w' to perform the strip command.
  chmod +w ${TMPDIR}/sdk/include/tensorflow/libtensorflow_cc.so
  chmod +w ${TMPDIR}/sdk/include/tensorflow/libtensorflow_framework.so.1
  strip ${TMPDIR}/sdk/include/tensorflow/libtensorflow_cc.so
  strip ${TMPDIR}/sdk/include/tensorflow/libtensorflow_framework.so.1
  mv ${TMPDIR}/sdk/include/tensorflow/libtensorflow_*.so* ${TMPDIR}/sdk/lib

  # third party packages doesn't ship with header files. Copy the headers
  # over so user defined ops can be compiled.
  mkdir -p ${TMPDIR}/sdk/include/google
  mkdir -p ${TMPDIR}/sdk/include/third_party
  pushd ${RUNFILES%org_tensorflow}/com_google_protobuf/src/google
  for header in $(find protobuf -name \*.h); do
    mkdir -p "${TMPDIR}/sdk/include/google/$(dirname ${header})"
    cp -L "$header" "${TMPDIR}/sdk/include/google/$(dirname ${header})/"
  done
  popd
  cp -RL $RUNFILES/third_party/eigen3 ${TMPDIR}/sdk/include/third_party
  cp -RL ${RUNFILES%org_tensorflow}/eigen_archive/* ${TMPDIR}/sdk/include/
  cp -RL ${RUNFILES%org_tensorflow}/nsync/public/* ${TMPDIR}/sdk/include
  cp -L ${RUNFILES%org_tensorflow}/com_google_protobuf/protoc ${TMPDIR}/sdk/bin

  # package all files into the target file.
  pushd ${TMPDIR}
  rm -f MANIFEST
  echo $(date) : "=== Building sdk package"
  tar czvf tensorflow_sdk.tar.gz sdk/ 1> /dev/null
  popd
  mkdir -p ${DEST}
  mv ${TMPDIR}/tensorflow_sdk.tar.gz ${DEST}
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output sdk package file is: ${DEST}/tensorflow_sdk.tar.gz"
}

main "$@"

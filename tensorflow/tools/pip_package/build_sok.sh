# Before we leave the top-level directory, make sure we know how to
# call python.
if [[ -e tools/python_bin_path.sh ]]; then
source tools/python_bin_path.sh
fi

HOROVOD_NCCL_LINK=SHARED HOROVOD_GPU_OPERATIONS=NCCL pip install 'horovod>=0.26.1'
pip install scikit-build
pip install cmake==3.21.1
pip install twine

export ENABLE_DEEPREC=ON
export DeepRecWorkdir=`pwd`
export DeepRecBuild=`pwd`/bazel-DeepRec
export MAKEFLAGS=-j$(nproc)
cd ./bazel-DeepRec/external/hugectr/sparse_operation_kit

"${PYTHON_BIN_PATH:-python}" setup.py install

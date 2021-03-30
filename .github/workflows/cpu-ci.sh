#!/bin/bash

git --version
python --version
pip --version
gcc --version
g++ --version
bazel version

# Build Tensorflow
cd /root/pai-tensorflow

./configure \
&& bazel build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 \
    --copt=-O3 \
    --copt=-Wformat \
    --copt=-Wformat-security \
    --copt=-fstack-protector \
    --copt=-fPIC \
    --copt=-fpic \
    --linkopt=-znoexecstack \
    --linkopt=-zrelro \
    --linkopt=-znow \
    --linkopt=-fstack-protector \
    --copt="-Wno-sign-compare" \
    --copt="-march=native" \
    //tensorflow/tools/pip_package:build_pip_package \
&& mkdir -p ./wheels \
&& bazel-bin/tensorflow/tools/pip_package/build_pip_package ./wheels

# BladeDISC

## Description

BladeDISC is an end-to-end machine learning compiler open-sourced by Alibaba, which can be directly used in DeepRec. 

> BladeDISC project address: https://github.com/alibaba/BladeDISC.

At present, DeepRec and BladeDISC cannot be directly compiled from source code, and we will use this method in the future. We need to compile and generate the BladeDISC whl package, and import blade_disc in the user code to use. For the scenario of using C++ for serving, the serving framework needs to link the BladeDISC so. The steps are as follows.

## How To Enable BladeDISC

### 1. Compile DeepRec

For compilation instruction, please refer to [DeepRec-Compile-And-Install](https://deeprec.readthedocs.io/zh/latest/DeepRec-Compile-And-Install.html#). The generated whl package will be used when compiling BladeDISC.

Note: Currently, the versions of bazel required to compile DeepRec and BladeDISC are inconsistent (this is one of the reasons why direct source code compilation is currently not possible, and we will upgrade to the same version later), so we will use the virtualenv environment to compile BladeDISC below.

### 2. Compile BladeDISC

In the docker container that compiles DeepRec.

- Install the DeepRec whl package.

- Download BladeDISC source code.
  
```bash
git clone https://github.com/alibaba/BladeDISC.git
git checkout features/deeprec2208-cu114
git submodule update --init --recursive
```

- Configure the compilation environment.
  
```bash
# prepare venv
pip3 install virtualenv

python3 -m virtualenv /opt/venv_disc/

source /opt/venv_disc/bin/activate

pip3 install tensorflow-1.15.5+deeprec2208-cp36-cp36m-linux_x86_64.whl

# install bazel
cd BladeDISC
apt-get update
bash ./docker/scripts/install-bazel.sh
```

- compile BladeDISC
  
```bash
# configure
./scripts/python/tao_build.py /opt/venv_disc/ --compiler-gcc default --bridge-gcc default -s configure

# generate libtao_ops.so，path: tao/bazel-bin/libtao_ops.so
./scripts/python/tao_build.py /opt/venv_disc/ -s build_tao_bridge

# generate tao_compiler_main
# path: tf_community/bazel-bin/tensorflow/compiler/decoupling/tao_compiler_main 
./scripts/python/tao_build.py /opt/venv_disc/ -s build_tao_compiler

# generate disc whl package
cp tf_community/bazel-bin/tensorflow/compiler/decoupling/tao_compiler_main tao/python/blade_disc_tf
cp tao/bazel-bin/libtao_ops.so  tao/python/blade_disc_tf
cd tao
python3 setup.py bdist_wheel  
```

- Install BladeDISC  
```bash
pip install dist/blade_disc_tf1155-0.2.0-py3-none-any.whl
```

### 3. Use BladeDISC in python

In user code: 

```python
import blade_disc_tf as disc
disc.enable()
```

### 4. Use BladeDISC in c++ serving
The c++ serving code needs to link libtao_ops.so, and the following two environment variables need to be set to enable disc optimization:
```bash
export BRIDGE_ENABLE_TAO=true
export TAO_COMPILER_PATH=/path-to/tao_compiler_main
```

Taking tensorflow_serving as an example, we can specify the path of libtao_ops.so (-L/xxx/xxx/mylib/) through -L at compile time, or copy libtao_ops.so to the system lib path (for example: /usr/local/lib /), so that libtao_ops.so can be linked to tensorflow_serving.

Assuming that the compiled libtao_ops.so is located at: /xxx/libtao_ops.so, we need to modify it as follows:
```
apt-get update && apt-get install patchelf
patchelf --remove-needed libtensorflow_framework.so.1 /xxx/libtao_ops.so

# for runtime
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/xxx/
export LD_LIBRARY_PATH
```

```
apt-get update
apt-get install autotools-dev
apt-get install automake
apt-get install libtool
export TF_CUDA_COMPUTE_CAPABILITIES="7.0,7.5,8.0"
```

The modification in tensorflow_serving is as follows: tensorflow_serving/model_servers/BUILD
```
 cc_binary(
     name = "tensorflow_model_server_main_lib",
     ...
     deps = [
         ...
         "@org_tensorflow//tensorflow/core/platform/hadoop:hadoop_file_system",
         "@org_tensorflow//tensorflow/core/platform/s3:s3_file_system",
+        "@org_tensorflow//tensorflow/stream_executor",
+        "@org_tensorflow//tensorflow/stream_executor:stream_executor_impl",
+        "@org_tensorflow//tensorflow/stream_executor:stream_executor_internal",
+        "@org_tensorflow//tensorflow/stream_executor:stream_executor_pimpl",
+        "@org_tensorflow//tensorflow/stream_executor:kernel_spec",
+        "@org_tensorflow//tensorflow/stream_executor:kernel",
+        "@org_tensorflow//tensorflow/stream_executor:scratch_allocator",
+        "@org_tensorflow//tensorflow/stream_executor:timer",
+        "@org_tensorflow//tensorflow/stream_executor/host:host_platform",
+    ],
+    linkopts = [
+        "-ltao_ops -L/xxx/",
+        "-Wl,-no-as-needed",
     ],
     ...
```

At the same time, because the above BUILD file relies on stream_executor, and the current visiablity of stream_executor is "friends", not "public", here we need to modify the DeepRec file ./tensorflow/stream_executor/BUILD referenced by tensorflow_serving as follows:
```
package(
-    default_visibility = [":friends"],
+    default_visibility = ["//visibility:public"],
     licenses = ["notice"],  # Apache 2.0
)
```

When BladeDISC is compiled with DeepRec, GLIBCXX_USE_CXX11_ABI=1 is used by default, while tensorflow_serving uses GLIBCXX_USE_CXX11_ABI=0 by default, so both sides need to be unified. This document takes modifying the .bazelrc file of tensorflow_serving as an example:
```
- build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0
+ build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=1
```

tensorflow_serving compilation script:
```
bazel build -c opt --config=cuda tensorflow_serving/...
```

tensorflow_serving compilation: [tfs compilation](https://deeprec.readthedocs.io/zh/latest/TFServing-Compile-And-Install.html)


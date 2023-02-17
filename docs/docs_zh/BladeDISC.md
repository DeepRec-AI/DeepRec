# BladeDISC
BladeDISC是阿里巴巴开源的端到端机器学习编译器，本文档主要介绍BladeDISC在中的DeepRec使用。BladeDISC开源项目地址: https://github.com/alibaba/BladeDISC .

目前DeepRec和BladeDISC暂时不能通过源码直接编译，后续我们会重构到使用此方式。目前我们需要通过编译生成BladeDISC whl包，并且在用户代码中import blade_disc来使用。对于使用C++进行serving的场景，serving框架需要link生成的BladeDISC的so。具体的步骤如下。

## DeepRec编译
```python
sudo nvidia-docker run -it --name=deeprec --net=host --gpus all  -v /home/workspace:/home/workspace registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-gpu-py36-cu110-ubuntu18.04 bash
```
具体编译步骤见：[DeepRec-Compile-And-Install](https://deeprec.readthedocs.io/zh/latest/DeepRec-Compile-And-Install.html#)，生成whl包。我们需要将deeprec whl安装在docker中，BladeDISC的编译依赖安装好的deeprec。

注意：目前编译DeepRec和BladeDISC需要的bazel版本不一致(这也是目前不能直接源码编译的原因之一，后续我们会升级到相同版本)，所以下面编译BladeDISC，我们使用virtualenv环境。

## BladeDISC编译
编译步骤如下:

- 安装生成的DeepRec whl包
- clone BladeDISC代码
```
git clone https://github.com/alibaba/BladeDISC.git
git checkout features/deeprec2208-cu114
git submodule update --init --recursive
```

- 安装编译环境
```
# prepare venv
pip3 install virtualenv

python3 -m virtualenv /opt/venv_disc/

source /opt/venv_disc/bin/activate

# 安装上面编译出来的whl包
pip3 install tensorflow-1.15.5+deeprec2208-cp36-cp36m-linux_x86_64.whl

# 安装bazel
cd BladeDISC
apt-get update
bash ./docker/scripts/install-bazel.sh
```

- 编译BladeDISC
```
# configure
./scripts/python/tao_build.py /opt/venv_disc/ --compiler-gcc default --bridge-gcc default -s configure

# 生成libtao_ops.so，生成路径是tao/bazel-bin/libtao_ops.so
./scripts/python/tao_build.py /opt/venv_disc/ -s build_tao_bridge

# 生成tao_compiler_main
# 生成路径是tf_community/bazel-bin/tensorflow/compiler/decoupling/tao_compiler_main 
./scripts/python/tao_build.py /opt/venv_disc/ -s build_tao_compiler

# 生成disc whl包
cp tf_community/bazel-bin/tensorflow/compiler/decoupling/tao_compiler_main tao/python/blade_disc_tf
cp tao/bazel-bin/libtao_ops.so  tao/python/blade_disc_tf
cd tao
python3 setup.py bdist_wheel
```

编译后的whl包在dist目录下。

- 安装生成的whl包
```
pip install dist/blade_disc_tf1155-0.2.0-py3-none-any.whl
```

## python使用方式
在代码中增加下面代码来enable disc，
```
import blade_disc_tf as disc
disc.enable()
```

## c++推理使用方式
c++推理代码在编译时需要链接libtao_ops.so，并且在执行时需要设置以下两个环境变量打开disc优化：
```
export BRIDGE_ENABLE_TAO=true
export TAO_COMPILER_PATH=/path-to/tao_compiler_main
```

以tensorflow_serving为例，我们可以在编译时通过-L指定libtao_ops.so所在路径(-L/xxx/xxx/mylib/)，或者拷贝libtao_ops.so到系统lib路径下(例如：/usr/local/lib/)，这样能将libtao_ops.so链接到tensorflow_serving中。

假设编译出来的libtao_ops.so位置在：/xxx/libtao_ops.so ，对于so要做一些处理如下：
```
apt-get update && apt-get install patchelf
patchelf --remove-needed libtensorflow_framework.so.1 /xxx/libtao_ops.so

# 下面为运行期准备
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/xxx/
export LD_LIBRARY_PATH
```

一些环境准备
```
apt-get update
apt-get install autotools-dev
apt-get install automake
apt-get install libtool
export TF_CUDA_COMPUTE_CAPABILITIES="7.0,7.5,8.0"
```

在tensorflow_serving中的修改如下：tensorflow_serving/model_servers/BUILD
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

同时由于上面BUILD文件中引入了stream_executor，而stream_executor目前的visiablity是"friends"，不是"public"，这里我们需要将tensorflow_serving引用的DeepRec文件./tensorflow/stream_executor/BUILD修改如下：
```
package(
-    default_visibility = [":friends"],
+    default_visibility = ["//visibility:public"],
     licenses = ["notice"],  # Apache 2.0
)
```

BladeDISC引用DeepRec编译时，默认GLIBCXX_USE_CXX11_ABI=1，而tensorflow_serving默认GLIBCXX_USE_CXX11_ABI=0，所以两边需要统一。本文档以修改tensorflow_serving的.bazelrc文件为例：
```
- build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0
+ build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=1
```

最终编译命令：
```
bazel build -c opt --config=cuda tensorflow_serving/...
```

tensorflow_serving具体编译详见：[tfs编译](https://deeprec.readthedocs.io/zh/latest/TFServing-Compile-And-Install.html)


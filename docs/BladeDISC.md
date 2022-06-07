# BladeDISC
BladeDISC是阿里巴巴开源的端到端机器学习编译器，它可以在DeepRec中直接使用。开源项目地址: https://github.com/alibaba/BladeDISC .

目前DeepRec是通过编译生成BladeDISC whl包，并且在用户代码中import blade_disc来使用。后续我们会在DeepRec代码直接编译BladeDISC源码，这样用户使用更加方便。

## DeepRec编译
```python
sudo nvidia-docker run -it --name=xxx --net=host --gpus all  -v /home/workspace:/home/workspace registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-gpu-py36-cu110-ubuntu18.04 bash
```
具体编译步骤见：[https://github.com/alibaba/DeepRec#how-to-build](https://github.com/alibaba/DeepRec#how-to-build)，生成whl包，需要安装在下面BladeDISC所在的docker中。

注意：这里使用两个docker的原因是，BladeDISC需要的bazel版本和DeepRec不一致，所以使用两个docker以示区别，两个docker镜像都是DeepRec的镜像，只是BladeDISC的需要安装高版本的Bazel(在下面有详叙)。或者在同一个docker中通过切换bazel版本的方式进行编译也是很方便的。

## BladeDISC编译
```python
sudo nvidia-docker run -it --name=xxx --net=host --gpus all  -v /home/workspace:/home/workspace registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-gpu-py36-cu110-ubuntu18.04 bash
```
编译步骤如下：

- 安装生成的DeepRec whl包
- clone代码
```
git clone https://github.com/alibaba/BladeDISC.git
git submodule update --init --recursive
```

- 安装编译环境(bazel + cmake)
```
wget https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-installer-linux-x86_64.sh
sh bazel-5.0.0-installer-linux-x86_64.sh
rm -rf /home/pai/bin/bazel 
ln -s /usr/local/lib/bazel/bin/bazel /home/pai/bin/bazel


wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.sh
mv cmake-3.20.0-linux-x86_64.sh /tmp/cmake-install.sh
chmod u+x /tmp/cmake-install.sh
mkdir -p /opt/cmake
/tmp/cmake-install.sh --skip-license --prefix=/opt/cmake
export PATH=/opt/cmake/bin:$PATH
```

- 编译BladeDISC
```
cd scripts/python
./tao_build.py /home/pai -s configure --bridge-gcc=7.5 --compiler-gcc=7.5
./tao_build.py /home/pai -s build_tao_compiler
./tao_build.py /home/pai -s build_tao_bridge
cd ../..
cp tf_community/bazel-bin/tensorflow/compiler/decoupling/tao_compiler_main tao/python/blade_disc_tf
cp  tao/bazel-bin/libtao_ops.so  tao/python/blade_disc_tf
cd tao
python3 setup.py bdist_wheel
```
编译后的whl包在dist目录下。

- 安装生成的whl包
```
pip install dist/blade_disc_gpu_tf1155-0.1.0-py3-none-any.whl
```

## BladeDISC使用方式
在代码中增加下面代码来enable disc，
```
import blade_disc_tf as disc
disc.enable()
```


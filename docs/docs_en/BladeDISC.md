# BladeDISC

## Description

BladeDISC is an end-to-end machine learning compiler open-sourced by Alibaba, which can be directly used in DeepRec. 

> BladeDISC project address: https://github.com/alibaba/BladeDISC.

At present,  users need to manually generate the BladeDISC whl package and call BladeDISC explicitly in the code. In the future, we will integrate BladeDISC into the DeepRec code to make it more convenient for users to use.

## How To Enable BladeDISC

### 1. Compile DeepRec

For compilation instruction, please refer to [https://github.com/alibaba/DeepRec#how-to-build](https://github.com/alibaba/DeepRec#how-to-build). The generated whl package will be used when compiling BladeDISC.

### 2. Compile BladeDISC

In the docker container that compiles DeepRec.

- Install the DeepRec whl package.

- Download BladeDISC source code.
  
  ```bash
  git clone https://github.com/alibaba/BladeDISC.git
  git submodule update --init --recursive
  ```

- Configure the compilation environment.
  
  ```bash
  # Update bazel
  wget https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-installer-linux-x86_64.sh
  sh bazel-5.0.0-installer-linux-x86_64.sh
  rm -rf /home/pai/bin/bazel 
  ln -s /usr/local/lib/bazel/bin/bazel /home/pai/bin/bazel
  
  # Install cmake
  wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.sh
  mv cmake-3.20.0-linux-x86_64.sh /tmp/cmake-install.sh
  chmod u+x /tmp/cmake-install.sh
  mkdir -p /opt/cmake
  /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake
  export PATH=/opt/cmake/bin:$PATH
  ```

- compile BladeDISC
  
  ```bash
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

- Install BladeDISC
  
  ```bash
  pip install dist/blade_disc_gpu_tf1155-0.1.0-py3-none-any.whl
  ```

### 3. How to use BladeDISC

In user code: 

```python
import blade_disc_tf as disc
disc.enable()
```

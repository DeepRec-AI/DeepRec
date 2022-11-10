# Embedding on PMEM
## 将Embedding Variable存到PMEM的好处
DeepRec拥有超大规模分布式训练能力，支持万亿样本模型训练和千亿Embedding Processing。稀疏模型中90%以上的数据是Embedding Variable，用于具有超大规模特征的大规模训练。在这种情况下，内存容量成为瓶颈之一。将Embedding Variable存到 PMEM 将带来以下好处：
1. 提高大规模分布式训练的内存存储能力；
2. 降低 TCO；
## 将PMEM配置成内存模式来保存Embedding Variable
通过开源程序ipmctl将物理机上的持久内存配置成百分之百内存模式：
```bash
# ipmctl create -goal memorymode=100
```
- 注：在虚拟机客户机实例（如re7p）里无法将物理机的PMEM配置成内存模式。
重启后系统的可用内存就为所有的PMEM大小，这时Embedding Variable就会存到PMEM中。
## 将PMEM配置成应用直接访问FSDAX模式来保存Embedding Variable
### 裸金属上配置PMEM为FSDAX模式：
```bash
# ipmctl create -goal persistentmemorytype=appdirect
# reboot
# ndctl create-namespace --region region0 --mode fsdax
# mkfs.ext4 /dev/pmem0
# mount -o dax /dev/pmem0 /mnt/pmem0
```
配置完，请检查FSDAX模式是否设置成功：
```bash
# ipmctl show -memoryresources
 MemoryType   | DDR         | PMemModule   | Total
==========================================================
 Volatile     | 256.000 GiB | 0.000 GiB    | 256.000 GiB
 AppDirect    | -           | 1008.000 GiB | 1008.000 GiB
 Cache        | 0.000 GiB   | -            | 0.000 GiB
 Inaccessible | 0.000 GiB   | 5.937 GiB    | 5.937 GiB
 Physical     | 256.000 GiB | 1013.937 GiB | 1269.937 GiB
# ndctl list -NR
{
  "regions":[
    {
      "dev":"region0",
      "size":1082331758592,
      "available_size":0,
      "max_available_extent":0,
      "type":"pmem",
      "iset_id":9218623383794094352,
      "persistence_domain":"memory_controller",
      "namespaces":[
        {
          "dev":"namespace0.0",
          "mode":"fsdax",
          "map":"dev",
          "size":1065418227712,
          "uuid":"c5c8759c-abb8-4f75-a402-2bbdba76ebf0",
          "sector_size":512,
          "align":2097152,
          "blockdev":"pmem0"
        }
      ]
    }
  ]
}
```
### 阿里云虚拟机实例re7p上配置Host PMEM为FSDAX模式：
以下命令以实例规格ecs.re7p.16xlarge（https://help.aliyun.com/document_detail/25378.html?spm=a2c4g.11186623.6.605.68ec600d0TJFNo#re7p） 为例。
```bash
# vi /etc/default/grub
在最后加上以下一行：
GRUB_CMDLINE_LINUX="memmap=1008G!257G"
# sudo grub2-mkconfig -o /boot/grub2/grub.cfg
# reboot
# sudo mkfs.ext4 /dev/pmem0
# mkdir /mnt/pmem0
# mount -o dax /dev/pmem0 /mnt/pmem0
```
### FSDAX模式下在Docker 容器里编译和安装DeepRec:
```bash
# docker pull registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-cpu-py36-ubuntu18.04
# git clone https://github.com/alibaba/DeepRec.git
# git clone https://github.com/memkind/memkind.git
# docker run -it -name test  -v /host_code_path:/work  -v /mnt/pmem0:/mnt/pmem0 --privileged registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-cpu-py36-ubuntu18.04 /bin/bash
# apt update
# apt install libpmem-dev gzip numactl gdb autoconf -y
# pip install pandas
# pip install numpy==1.16.0
# cd /root/memkind/
# ./autogen.sh && ./configure
# make clean;make -j;make install
# cd /work/DeepRec
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# ./configure
Do you wish to build TensorFlow with PMEM support? [y/N]: y
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --copt="-L/usr/local/lib" --copt="-lpmem" --copt="-lmemkind"  --config=opt //tensorflow/tools/pip_package:build_pip_package
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# pip3 install /tmp/tensorflow_pkg/tensorflow-1.15.5+deeprec2201-cp36-cp36m-linux_x86_64.whl
```
## 将PMEM配置成NUMA节点来保存Embedding Variable
### 裸金属上配置PMEM为NUMA节点：

请安装v66以上版本的ndctl和 daxctl，在PMEM上创建devdax模式的命名空间，将持久内存从devdax模式重新配置成system-ram模式，如果有2个socket，请对于socket 1上的PMEM执行类似的操作。
```bash
# ndctl create-namespace --mode=devdax --map=mem
# daxctl reconfigure-device --mode=system-ram --region=0 dax0.0
```
在此操作之后，持久内存被配置为一个单独的 NUMA 节点，并且可以用作易失性内存。
```bash
# numactl -H
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
node 0 size: 191904 MB
node 0 free: 109899 MB
node 1 cpus:
node 1 size: 759808 MB
node 1 free: 759807 MB
node distances:
node   0   1
  0:  10  17
  1:  17  10
```
### 阿里云虚拟机实例re7p上配置Host PMEM为NUMA节点：
以下命令以实例规格ecs.re7p.16xlarge（https://help.aliyun.com/document_detail/25378.html?spm=a2c4g.11186623.6.605.68ec600d0TJFNo#re7p） 为例。
```bash
# vi /etc/default/grub
删去以下一行：
GRUB_CMDLINE_LINUX="memmap=1008G!257G"
# sudo grub2-mkconfig -o /boot/grub2/grub.cfg
# reboot
[root@iZ2zei09caif72ul6x3iaiZ ~]# numactl -H
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
node 0 size: 249568 MB
node 0 free: 247930 MB
node 1 cpus:
node 1 size: 1016062 MB
node 1 free: 1015842 MB
```
### KMEM DAX模式下在Docker 容器里编译和安装DeepRec:
```bash
# docker pull registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-cpu-py36-ubuntu18.04
# git clone https://github.com/alibaba/DeepRec.git
# git clone https://github.com/memkind/memkind.git
# docker run -i -t  -v /root:/root --privileged registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-cpu-py36-ubuntu18.04 /bin/bash
# apt update
# apt install libpmem-dev gzip numactl gdb autoconf -y
# pip install pandas
# pip install numpy==1.16.0
# cd /root/memkind/
# ./autogen.sh && ./configure
# make clean;make -j;make install
# cd /work/DeepRec
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# export MEMKIND_DAX_KMEM_NODES=1
# ./configure
Do you wish to build TensorFlow with PMEM support? [y/N]: y
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --copt="-L/usr/local/lib" --copt="-lmemkind"  --config=opt //tensorflow/tools/pip_package:build_pip_package
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# pip3 install /tmp/tensorflow_pkg/tensorflow-1.15.5+deeprec2201-cp36-cp36m-linux_x86_64.whl
```
## 将Embedding Variable存到PMEM上验证WDL模型性能
### 在PMEM内存模式上运行WDL Stand-alone Training
```bash
1. 用户通过命令设置WDL模型Embedding Variable的存储类型为DRAM；
2. 执行WDL train过程。
```
### 在PMEM FSDAX模式上运行WDL Stand-alone Training
```bash
1. 用户通过命令设置WDL模型Embedding Variable的存储类型为PMEM_LIBPMEM，设置存储路径指向mount的持久内存目录，设置持久内存上存储数据占用空间大小；
2. 执行WDL train过程。
```
### 在PMEM KMEM DAX模式上运行WDL Stand-alone Training
```bash
1. 用户通过命令设置WDL模型Embedding Variable的存储类型为PMEM_MEMKIND；
2. 执行WDL train过程。
```

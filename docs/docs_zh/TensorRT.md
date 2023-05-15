# TensorRT

TensorRT是NVIDIA公司推出的一个高性能深度学习推理引擎，它可以将深度学习模型优化为高效的推理引擎，从而在生产环境中实现快速、低延迟的推理。为了方便用户同时使用DeepRec和TensorRT，我们将TensorRT集成进DeepRec，DeepRec会对用户的graph进行分析，将TensorRT能识别的subgraph进行clustering，这里类似XLA的圈图方式。对于每个clustering子图，使用一个TRTEngineOp驱动执行。

## 环境配置
目前在DeepRec发布的镜像中暂时没有安装TensorRT环境，需要用户手动安装，下一次的镜像发布会带上安装好的环境。
可以自己下载TRT或者使用我们提供的[TRT-8.4.2.4](http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/tensorrt/TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz)

解压tar包并且拷贝到合适的位置，譬如：/usr/local/TensorRT-8.4.2.4，我们需要设置环境变量如下：
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/TensorRT-8.4.2.4/lib
export TENSORRT_INSTALL_PATH=/usr/local/TensorRT-8.4.2.4
```

## 生成saved_model
用户需要生成自己的saved model，文档中以"modelzoo/features/embedding_variable/wide_and_deep/train.py"为示例，用户首先需要生成saved model，
```python
python train.py --saved_model_path=./pb
```

## 转换模型
对于生成的模型转换成TRT模型，这里需要使用"tensorflow/compiler/tf2tensorrt/tool/tf2trt.py"脚本进行转换。
```python
python tf2trt.py
```
注意tf2trt.py文件中有一些参数需要自己配置一下，
```
...
if __name__ == "__main__":
  run_params = {
    # 模型精度要求，默认是FP32，
    # TensorRT 提供了 FP16 量化与 INT8 量化。
    'precision_mode':"FP32",
    # 是否生成将在运行时构建TRT network和engine的动态TRT ops。
    'dynamic_engine': True,
    # 对于INT8 量化，TRT需要进行校准(calibration)，生成校准文件。
    'use_calibration': False,
    # input数据的最大batch size
    'max_batch_size': 1024,
    # 在线转换还是离线转换
    'convert_online': False,
    # 模型是否使用了embedding variable
    'use_ev': True,
  }

  # 原始的模型saved model所在位置
  saved_model_dir = '/model/pb'
  # 转换之后的模型保存位置
  trt_saved_model_dir = './trtmodel'

  ConvertGraph(run_params, saved_model_dir, trt_saved_model_dir)
...
```

## 执行模型
使用[DeepRec-AI/serving](https://github.com/DeepRec-AI/serving)加载新生成的模型。注意需要重新编译serving，并且在此次编译时需要配置环境变量 `export TF_NEED_TENSORRT=1`,这样保证在serving so的符号表中是包含了TRT符号的。


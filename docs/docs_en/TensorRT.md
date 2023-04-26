# TensorRT

TensorRT is a high-performance deep learning inference engine launched by NVIDIA. It can optimize the deep learning model into an efficient inference engine, so as to achieve fast and low-latency inference in the production environment. In order to facilitate users to use DeepRec and TensorRT at the same time, we integrate TensorRT into DeepRec. DeepRec will analyze the user's graph and cluster the subgraphs that TensorRT can recognize. This is similar to the XLA clustering graph method. For each clustering subgraph, use a TRT EngineOp drive it to execute.

## Environment configuration
At present, the TensorRT environment is not installed in the image released by DeepRec for the time being, and the user needs to install it manually. The next image release will bring the installed environment.
You can download TRT yourself or use the tar we offered:[TRT-8.4.2.4](http://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/tensorrt/TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz)

Unzip the tar package and copy it to a suitable location, for example: /usr/local/TensorRT-8.4.2.4, and then we need to set the environment variables as follows:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/TensorRT-8.4.2.4/lib
export TENSORRT_INSTALL_PATH=/usr/local/TensorRT-8.4.2.4
```

## generate saved_model
Users need to generate their own saved_model. The document takes "modelzoo/features/embedding_variable/wide_and_deep/train.py" as an example. Users first need to generate a saved model:
```python
python train.py --saved_model_path=./pb
```

## convert model
To convert the generated model into a TRT model, you need to use the "tensorflow/compiler/tf2tensorrt/tool/tf2trt.py" script for conversion.
```python
python tf2trt.py
```
Note that there are some parameters in the tf2trt.py file that need to be configured by yourself.
```
...
if __name__ == "__main__":
  run_params = {
    # Model accuracy requirements, the default is FP32,
    # TensorRT provides FP16 quantization and INT8 quantization.
    'precision_mode':"FP32",

    # Whether to generate dynamic TRT ops that will
    # build the TRT network and engine at runtime.
    'dynamic_engine': True,

    # For INT8 quantization, TRT needs to be calibrated
    # to generate a calibration file.
    'use_calibration': False,

    # max size for the input batch.
    'max_batch_size': 1024,

    # convert model online or offline
    'convert_online': False,

    # User model use EmbeddingVariable or not. 
    'use_ev': True,
  }

  # the path to load the SavedModel.
  saved_model_dir = '/model/pb'
  # The converted model save location.
  trt_saved_model_dir = './trtmodel'

  ConvertGraph(run_params, saved_model_dir, trt_saved_model_dir)
...
```

## Run model
Use [DeepRec-AI/serving](https://github.com/DeepRec-AI/serving) to load new saved_model。Note that serving needs to be recompiled, and the environment variable `export TF_NEED_TENSORRT=1` needs to be configured before this compilation, so as to ensure that the symbol table of serving so contains the TRT symbol.


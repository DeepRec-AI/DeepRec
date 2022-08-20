# DeepRec低精度优化工具
## 功能介绍
该工具能够对DeepRec推荐类模型进行CPU上的FP16、BF16和INT8优化，主要针对模型中的Embedding和Dense计算节点，支持自定义优化层、指定数值类型等操作。

## 使用方式
目前支持的输入、输出模型类型均为saved_model，通过low_precision_optimize.optimize接口进行优化。
```python
optimize(model_path, save_path, opt_config=None, data_type='BF16', calib_file=None)
```
具体参数说明如下：

| 参数 | 类型 | 说明 |
| ---  | ---  | ---  |
| model_path | string | 输入的原始模型存储路径 |
| save_path | string | 输出的优化模型存储路径 |
| opt_config | dict | 自定义的优化设置，key为指定的优化节点，value为指定的该节点的优化数值类型（"BF16"、"FP16"或"INT8"），该参数可缺省，即默认优化模型中所有可优化节点 |
| data_type | string | 用于指定优化的数据类型："FP16"、"BF16"或"INT8"，默认为"BF16"，该参数仅在opt_config参数缺省时生效，即将优化模型中所有可优化节点优化为指定的data_type类型 |
| calib_data | list | 用于INT8量化环节的校正数据集，该工具目前仅支持离线量化（即静态量化），故选择INT8优化时需给定校正数据集，校正数据集是一个包含若干组feed_dict的列表（具体准备方式可参考下方示例）。|

注：优化结束后会输出具体的优化节点信息（示例如下），可供参考以调整优化设置。
```shell
Optmization Result:
Optimize embedding to INT8: input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights
Optimize dense op to BF16: mlp_bot_layer/mlp_bot_hiddenlayer_1/MatMul
```
## 优化示例
具体示例如下：
```python
from low_precision_optimize import optimize

model_path = 'dlrm/saved_model'
save_path = 'dlrm/saved_model_opt'
calib_file = 'dlrm/calib_data.npy'

# 将模型中所有可优化节点优化为BF16类型
optimize(model_path, save_path, data_type='BF16')

# 将模型中所有可优化节点优化为FP16类型
optimize(model_path, save_path, data_type='FP16')

# 将模型中所有可优化节点优化为INT8类型
optimize(model_path, save_path, data_type='INT8', calib_file=calib_file)

# 指定具体优化设置
opt_dict={
    'input_layer/sparse_input_layer/input_layer/C1_embedding/embedding_weights': 'INT8',
    'mlp_top_layer/mlp_top_hiddenlayer_1/MatMul': 'BF16',
}
optimize(model_path, save_path, opt_dict, data_type='INT8', calib_file=calib_file)
```

校正数据集（calib_file）准备方式示例如下。
```python
import numpy as np

calib_data = list()
for i in range(10):
    feed_dict = {'features:0': '0,52,29,10,18434,48,5,10,24,0,1,0,10'}
    calib_data.append(feed_dict)
with open('calib_data.npy', 'wb') as f:
    np.save(f, calib_data)
```

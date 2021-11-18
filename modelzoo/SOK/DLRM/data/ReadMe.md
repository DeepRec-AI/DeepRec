## Dataset
### Prepare Dataset
Put data file **train.csv & eval.csv** into ./data/    

[Criteo Terabytes Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) will be used. Download these files. And there are several options for you to generate datasets.

#### [Option1] ####
Follow [TensorFlow's instructions](https://github.com/tensorflow/models/tree/master/official/recommendation/ranking/preprocessing) to process these files and save as CSV files.

#### [Option2] ####
Follow [HugeCTR's instructions](https://github.com/NVIDIA/HugeCTR/tree/master/samples/dlrm#preprocess-the-terabyte-click-logs) to process these files. Then convert the generated binary files to CSV files.
```shell
$ python3 bin2csv.py \
    --input_file="YourBinaryFilePath/train.bin" \
    --num_output_files=1024 \
    --output_path="./train/" \
    --save_prefix="train_"
```
```shell
$ python3 bin2csv.py \
    --input_file="YourBinaryFilePath/test.bin" \
    --num_output_files=64 \
    --output_path="./test/" \
    --save_prefix="test_"
```

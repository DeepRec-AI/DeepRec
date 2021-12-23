## Dataset
### Prepare Dataset    

[Criteo Terabytes Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) will be used. Download these files. And there are several options for you to generate datasets.

Follow [HugeCTR's instructions](https://github.com/NVIDIA/HugeCTR/tree/master/samples/dlrm#preprocess-the-terabyte-click-logs) to process these files. Then convert the generated binary files to CSV files.
```shell
$ python3 bin2bin.py \
    --input="YourBinaryFilePath/train.bin" \
    --output="./data/train/" 
```
```shell
$ python3 bin2bin.py \
    --input="YourBinaryFilePath/test.bin" \
    --output="./data/test/"
```

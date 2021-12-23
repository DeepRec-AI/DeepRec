# SparseOperationKit demo notebooks #
This directory contains a set of Jupyter Notebook demos for SparseOperationKit.

## Prerequisites ##
Before trying the notebooks here, you have to follow [these instructions](../../notebooks/README.md#Quickstart) to prepare the operating environment. Summarized bellow:
+ Pull the NGC Docker
```shell
$ docker pull nvcr.io/nvidia/merlin/merlin-tensorflow-training:21.11
```
+ Clone the HugeCTR Repo
```shell
$ git clone https://github.com/NVIDIA/HugeCTR hugectr
```
+ Start the Jupyter Notebook
    - Launch the container in interactive mode and mount the HugeCTR root directory into the container for your convenience by running this command:
    ```shell
    $ docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr -p 8888:8888 nvcr.io/nvidia/merlin/merlin-tensorflow-training:21.11
    ```
    - Start Jupyter using these commands:
    ```shell
    $ cd /hugectr/sparse_operation_kit/notebooks
    $ jupyter-notebook --allow-root --ip 0.0.0.0 --port 8888 --NotebookApp.token=’sok’
    ```
    - Connect to your host machine using port 8888 from your web browser: `http://[host machine]:8888`

## Notebook List ##
The notebooks are located within the container and can be found here: `/hugectr/sparse_operation_kit/notebooks`.

- [sparse_operation_kit_demo.ipynb](sparse_operation_kit_demo.ipynb): Demos of new TensorFlow plugins for sparse operations (currently, only embedding layers).
- [benchmark.ipynb](benchmark.ipynb): Benchmarking of the SparseOperationKit.
# DeepRec Processor Go Example

## 要求

- Go
- Protobuf v3.6.1


## 使用

1. 根据proto文件生成对应go文件

    ```sh
    mkdir tensorflow_eas
    go mod init demo
    ```

    根据 [Protobuf Docs](https://github.com/protocolbuffers/protobuf/tree/48cb18e5c419ddd23d9badcfe4e9df7bde1979b2#protocol-compiler-installation) 安装 protocol compiler

    安装 protocol compiler plugins for Go
    ```sh
    go install github.com/golang/protobuf/protoc-gen-go@v1.2.0
    ```

    添加$GOPATH/bin到PATH环境变量中

    根据proto文件生成go文件
    ```sh
    protoc --go_out=../go/tensorflow_eas -I../../processor/serving/ predict.proto
    ```

    注意：predict.proto文件位于DeepRec/serving/processor/serving

2. 安装protobuf go module

    ```sh
    go get github.com/golang/protobuf/proto@v1.2.0
    ```

3. 生成 DeepRec Serving Processor

    需要```libserving_processor.so```
    编译详见[https://github.com/alibaba/DeepRec](https://github.com/alibaba/DeepRec)项目首页“How to Build serving library”部分。

4. 生成 demo checkpoint 和 savedmodel

    ```sh
    python simple_model.py --saved_model_dir=xxx --checkpoint_dir=xxx
    ```
    如果没有设置saved_model_dir，默认路径为 '/tmp/saved_model'，
    如果没有设置checkpoint_dir，默认路径为 '/tmp/checkpoint/1'。
    ```sh
    python simple_model.py
    ```

    注意：simple_model.py文件位于DeepRec/serving/processor/tests/end2end/

5. 设置demo.go中的`saved_model_dir` and `checkpoint_dir`

    ```go
    var modelConfig = []byte(`{
        ...
        "checkpoint_dir": "/tmp/checkpoint/",
        "savedmodel_dir": "/tmp/saved_model/"
    } `)
    ```
    注意：这里的 checkpoint_dir 应该是 checkpoint dir 的父目录,
    比如 '/tmp/checkpoint/1'，设置 checkpoint_dir 为 '/tmp/checkpoint'。

6. 编译和运行

    修改demo.go中LDFLAGS为processor的路径
    ```sh
    go build .
    LD_PRELOAD=/path/to/libserving_processor.so ./demo
    ```

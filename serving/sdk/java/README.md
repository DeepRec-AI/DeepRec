# DeepRec Processor Java Example

## 要求

- Java Environment
- Protobuf v3.6.1


## 使用

1. 根据proto文件生成对应java文件

    ```sh
    mkdir bin lib
    ```

    lib目录下需要```protobuf-java.jar```，根据 [Protobuf Docs](https://github.com/protocolbuffers/protobuf/tree/48cb18e5c419ddd23d9badcfe4e9df7bde1979b2/java#build-from-source) 生成

    根据proto文件生成java文件
    ```sh
    protoc --java_out=../java -I../../processor/serving/ predict.proto
    ```

    注意：predict.proto文件位于DeepRec/serving/processor/serving

2. 下载JNA

    lib目录下需要```jna.jar```，访问 [JNA GitHub](https://github.com/java-native-access/jna) 下载

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

5. 设置Demo.java中的`saved_model_dir` and `checkpoint_dir`

    ```java
    public static String modelConfig =
        ... +
        "\"checkpoint_dir\": \"/tmp/checkpoint/\"," +
        "\"savedmodel_dir\": \"/tmp/saved_model/\"}";
    ```
    注意：这里的 checkpoint_dir 应该是 checkpoint dir 的父目录,
    比如 '/tmp/checkpoint/1'，设置 checkpoint_dir 为 '/tmp/checkpoint'。

6. 编译和运行

    ```sh
    javac -d bin/ src/main/java/Demo.java tensorflow/eas/Predict.java -cp lib/\*
    ```

    ```sh
    LD_PRELOAD=/path/to/libserving_processor.so java -cp bin/:lib/\* src.main.java.Demo tensorflow.eas.Predict
    ```

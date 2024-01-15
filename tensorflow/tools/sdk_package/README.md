Bazel rules and bash scripts to package the DeepRec C/C++ APIs and
runtime library into '\<DeepRec Root Path\>/tensorflow_sdk.tar.gz' archive.

## SDK Build

First of all, edit and run the configurating script **'./configure'** under
DeeRec root directory (supposed '\<DeepRec Root Path\>').

Then simply run the following commands under '\<DeepRec Root Path\>' to build
the DeepRec SDK package:

```sh
./build sdk
```
_This command will put the SDK package named 'tensorflow\_sdk.tar.gz' into
the directory below:_
>     <DeepRec Root Path>/built/sdk/[gpu|cpu]

## SDK usage:

To make use of DeepRec runtime SDK for C++ codes writting with original APIs
defined in TensorFlow, just decompress the SDK package into another work
directory (supposed '\<workdir path\>') with the command at first:

```sh
tar xzvf -C <workdir path> tensorflow_sdk.tar.gz
```

Then a directory named 'sdk' will be placed into the \<workdir path\>, which
contains necessary header files in the 'include' sub-directory, keeping the
original hierarchy in TensorFlow, and the 'libtensorflow_cc.so' dynamic
runtime library in the 'lib' sub-directoy to support TensorFlow running.

Just append **'-I\<workdir path\>/sdk/include'** to compiling arguments and
**'-L\<workdir path\>/sdk/lib'** -ltensorflow_cc to linking arguments, in the
cases of building a project, that contains codes using original TensorFlow
C++ APIs, together with DeepRec SDK.

Finally, to successfully run the binary building with DeepRec SDK, do not
forget to append '\<workdir path\>/sdk/lib' to **'LD_LIBRARY_PATH'** environment
variable.

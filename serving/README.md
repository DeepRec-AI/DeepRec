## How to build odl_processor so

1. configure bazel env, will generate .bazelrc file

   ./configure serving
   ./configure serving --mkl
   ./configure serving --mkl_open_source_v1_only
   ./configure serving --mkl_threadpool
   ./configure serving --mkl --cuda ...

   More details see: serving/tools/build/configure.py

2. build odl_processor library
  
   bazel build //serving/odl_processor/serving:libtf_processor.so
   bazel test -- //serving/odl_processor/... -//serving/odl_processor/framework:lookup_manual_test

3. Required libs:
   libiomp5.so  libmklml_intel.so  libstdc++.so.6  libtf_processor.so


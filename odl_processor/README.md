## How to build odl_processor so

1. start container pai-tf 1.12

2. configure bazel env

   sh configure.sh python [other params]
   sh configure.sh python3 [other params]

   Example:
     sh configure.sh python3 --mkl --cuda ...

3. build odl_processor library
  
   bazel build //odl_processor/serving:libtf_processor.so
   bazel test -- //odl_processor/... -//odl_processor/framework:lookup_manual_test

4. Required libs:
   libiomp5.so  libmklml_intel.so  libstdc++.so.6  libtf_processor.so


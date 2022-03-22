## How to build odl_processor so

1. start container pai-tf 1.12

2. configure bazel env

   sh configure.sh python

3. build odl_processor library
  
   bazel build //odl_processor/libodl_processor.so

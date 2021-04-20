rm -f .bazelrc

PYTHON=python3
if [[ "$#" -gt 0 ]]; then
  PYTHON=$1
fi

# python tools/build/configure.py --mkl
# python tools/build/configure.py --mkl_open_source_v1_only
# python tools/build/configure.py --mkl_threadpool
$PYTHON tools/build/configure.py "$@"


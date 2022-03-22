rm -f .bazelrc

PYTHON=python3
if [[ "$#" -gt 0 ]]; then
  PYTHON=$1
fi

$PYTHON tools/build/configure.py

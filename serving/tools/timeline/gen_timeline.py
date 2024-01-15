import sys
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline

def gen_timeline(src_name, dest_name):
  run_metadata = config_pb2.RunMetadata()
  with open(src_name, 'rb') as f:
    run_metadata.step_stats.ParseFromString(f.read())
  tl = timeline.Timeline(run_metadata.step_stats)
  content = tl.generate_chrome_trace_format()
  with open(dest_name, 'w') as f:
    f.write(content)

if __name__ == '__main__':
  # usage:
  # python gen_timeline.py timeline_file my_timeline.json
  #
  gen_timeline(sys.argv[1], sys.argv[2])


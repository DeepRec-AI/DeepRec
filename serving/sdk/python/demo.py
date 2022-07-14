import sys
import ctypes

from predict_pb2 import (
    ArrayDataType,
    ArrayShape,
    ArrayProto,
    PredictRequest,
    PredictResponse,
)

model_config = '{ \
    "omp_num_threads": 4, \
    "kmp_blocktime": 0, \
    "feature_store_type": "memory", \
    "serialize_protocol": "protobuf", \
    "inter_op_parallelism_threads": 10, \
    "intra_op_parallelism_threads": 10, \
    "init_timeout_minutes": 1, \
    "signature_name": "serving_default", \
    "read_thread_num": 3, \
    "update_thread_num": 2, \
    "model_store_type": "local", \
    "checkpoint_dir": "/tmp/checkpoint/", \
    "savedmodel_dir": "/tmp/saved_model/" \
}'

if __name__ == "__main__":
    # Load shared library
    processor = ctypes.cdll.LoadLibrary("libserving_processor.so")
    model_entry = ""
    state = ctypes.c_int(0)
    state_ptr = ctypes.pointer(state)
    processor.initialize.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    processor.initialize.restype = ctypes.POINTER(ctypes.c_char)
    model = processor.initialize(
        ctypes.create_string_buffer(model_entry.encode("utf-8")),
        ctypes.create_string_buffer(model_config.encode("utf-8")),
        state_ptr,
    )
    if state_ptr.contents == -1:
        print("initialize error", file=sys.stderr)

    # input type: float
    dtype = ArrayDataType.Value("DT_FLOAT")
    # input shape: [1, 1]
    array_shape = ArrayShape()
    array_shape.dim.append(1)
    array_shape.dim.append(1)
    # input array
    input = ArrayProto()
    input.float_val.append(1.0)
    input.dtype = dtype
    input.array_shape.CopyFrom(array_shape)
    # PredictRequest
    req = PredictRequest()
    req.signature_name = "serving_default"
    req.output_filter.append("y:0")
    req.inputs["x:0"].CopyFrom(input)
    buffer = req.SerializeToString()
    size = req.ByteSize()

    # do process
    output = ctypes.c_void_p(0)
    output_ptr = ctypes.pointer(output)
    output_size = ctypes.c_int(0)
    output_size_ptr = ctypes.pointer(output_size)
    processor.process.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    processor.process.restype = ctypes.c_int
    state = processor.process(model, buffer, size, output_ptr, output_size_ptr)

    # parse response
    output_string = ctypes.string_at(output, output_size)
    resp = PredictResponse()
    resp.ParseFromString(output_string)
    print(f"process returned state: {state}, response: {dict(resp.outputs.items())}")

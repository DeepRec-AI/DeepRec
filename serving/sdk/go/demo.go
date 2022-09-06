package main

/*
#cgo LDFLAGS: -ldl
#include <stdlib.h>
#include <dlfcn.h>

void* processor_initialize(void* f, const char* model_entry,
                    const char* model_config, int* state)
{
    void* (*initialize)(const char*, const char*, int*);

    initialize = (void* (*)(const char*, const char*, int*))f;
    return initialize(model_entry, model_config, state);
}

int processor_process(void* f, void* model_buf, const void* input_data,
            int input_size, void** output_data, int* output_size)
{
    int (*process)(void*, const void*, int, void**, int*);

    process = (int (*)(void*, const void*, int, void**, int*))f;
    return process(model_buf, input_data, input_size, output_data, output_size);
}
*/
import "C"
import (
	tensorflow_eas "demo/tensorflow_eas"
	errors "errors"
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	unsafe "unsafe"
)

var modelConfig = []byte(`{
    "omp_num_threads": 4,
    "kmp_blocktime": 0,
    "feature_store_type": "memory",
    "serialize_protocol": "protobuf",
    "inter_op_parallelism_threads": 10,
    "intra_op_parallelism_threads": 10,
    "init_timeout_minutes": 1,
    "signature_name": "serving_default",
    "read_thread_num": 3,
    "update_thread_num": 2,
    "model_store_type": "local",
    "checkpoint_dir": "/tmp/checkpoint/",
    "savedmodel_dir": "/tmp/saved_model/"
} `)

func main() {
	// Load shared library
	modelEntry := []byte(".")
	state := C.int(0)
	handle := C.dlopen(C.CString("libserving_process.so"), C.RTLD_LAZY)
	var err error
	if handle == nil {
		err = errors.New(C.GoString(C.dlerror()))
		println(err)
	}
	initialize := C.dlsym(handle, C.CString("initialize"))
	model := C.processor_initialize(initialize, (*C.char)(unsafe.Pointer(&modelEntry[0])), (*C.char)(unsafe.Pointer(&modelConfig[0])), &state)
	defer C.free(unsafe.Pointer(model))
	if int(state) == -1 {
		println("initialize error")
	}

	// input type: float
	dtype := tensorflow_eas.ArrayDataType_DT_FLOAT
	// input shape: [1, 1]
	var arrayShape tensorflow_eas.ArrayShape
	arrayShape.Dim = append(arrayShape.Dim, 1)
	arrayShape.Dim = append(arrayShape.Dim, 1)
	// input array
	var input tensorflow_eas.ArrayProto
	input.FloatVal = append(input.FloatVal, 1.0)
	input.Dtype = dtype
	input.ArrayShape = &arrayShape

	// Predictrequest
	var req tensorflow_eas.PredictRequest
	req.SignatureName = "serving_default"
	req.OutputFilter = append(req.OutputFilter, "y:0")
	req.Inputs = make(map[string]*tensorflow_eas.ArrayProto)
	req.Inputs["x:0"] = &input
	buffer, err := proto.Marshal(&req)
	if err != nil {
		println(err.Error())
	}
	size := C.int(proto.Size(&req))

	// do process
	output := unsafe.Pointer(nil)
	defer C.free(output)
	outputSize := C.int(0)
	process := C.dlsym(handle, C.CString("process"))
	state = C.processor_process(process, model, unsafe.Pointer(&buffer[0]), size, &output, &outputSize)

	// parse response
	outputString := C.GoBytes(output, outputSize)
	var resp tensorflow_eas.PredictResponse
	err = proto.Unmarshal(outputString, &resp)
	if err != nil {
		println(err.Error())
	}
	fmt.Printf("process returned state: %d, response: %s", int(state), resp.Outputs)

}

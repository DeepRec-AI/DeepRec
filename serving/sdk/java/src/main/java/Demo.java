package src.main.java;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import tensorflow.eas.Predict;

class Demo {

    public static String modelConfig =
        "{\"omp_num_threads\": 4," +
        "\"kmp_blocktime\": 0," +
        "\"feature_store_type\": \"memory\"," +
        "\"serialize_protocol\": \"protobuf\"," +
        "\"inter_op_parallelism_threads\": 10," +
        "\"intra_op_parallelism_threads\": 10," +
        "\"init_timeout_minutes\": 1," +
        "\"signature_name\": \"serving_default\"," +
        "\"read_thread_num\": 3," +
        "\"update_thread_num\": 2," +
        "\"model_store_type\": \"local\"," +
        "\"checkpoint_dir\": \"/tmp/checkpoint/\"," +
        "\"savedmodel_dir\": \"/tmp/saved_model/\"}";

    // Load shared library via JNA
    public interface Processor extends Library
    {
        Processor INSTANCE = (Processor) Native.load("serving_processor", Processor.class);

        // Define shared library function prototype
        public Pointer initialize(String modelEntry, String modelConfig, int[] state);

        public int process(Pointer model, byte[] buffer, int size, PointerByReference outputData, int[] outputSize);
    }

    public static void main(String[] args) {
        Demo demo = new Demo();
        String modelEntry = "";
        int[] state = {0};
        Pointer model = Processor.INSTANCE.initialize(modelEntry, modelConfig, state);
        if (state[0] == -1) {
            System.err.println("initialize error");
        }

        // input type: float
        Predict.ArrayDataType dtype = Predict.ArrayDataType.DT_FLOAT;
        // input shape: [1, 1]
        Predict.ArrayShape arrayShape =
            Predict.ArrayShape.newBuilder()
            .addDim(1)
            .addDim(1)
            .build();
        // input array
        Predict.ArrayProto input =
            Predict.ArrayProto.newBuilder()
            .addFloatVal((float) 1.0)
            .setDtype(dtype)
            .setArrayShape(arrayShape)
            .build();
        // PredictRequest
        Predict.PredictRequest req =
            Predict.PredictRequest.newBuilder()
            .setSignatureName("serving_default")
            .addOutputFilter("y:0")
            .putInputs("x:0", input)
            .build();
        byte[] buffer = req.toByteArray();
        int size = req.getSerializedSize();

        // do process
        PointerByReference output = new PointerByReference();
        int[] outputSize = {0};
        state[0] = Processor.INSTANCE.process(model, buffer, size, output, outputSize);

        // parse response
        byte[] outputString = output.getValue().getByteArray(0, outputSize[0]);
        String s = new String(outputString);
        try {
            Predict.PredictResponse resp =
                Predict.PredictResponse.newBuilder()
                .mergeFrom(outputString)
                .build();
            System.out.println(resp.toString());
        } catch (Exception e) {
            System.err.println("parse response error");
        }

    }
}

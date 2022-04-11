#ifndef SERVING_PROCESSOR_SERVING_TF_PROCESSOR_H
#define SERVING_PROCESSOR_SERVING_TF_PROCESSOR_H

extern "C" {
void* initialize(const char* model_entry, const char* model_config, int* state);
int process(void* model_buf, const void* input_data, int input_size,
            void** output_data, int* output_size);
int batch_process(void* model_buf, const void* input_data[], int* input_size,
                  void* output_data[], int* output_size);
int get_serving_model_info(void* model_buf, void** output_data, int* output_size);
}
#endif

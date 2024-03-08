#include <iostream>
#include<stdlib.h>
#include "serving/processor/serving/processor.h"
#include "serving/processor/serving/predict.pb.h"
#include<stdio.h>

static const char* model_config = "{ \
    \"omp_num_threads\": 4, \
    \"kmp_blocktime\": 0, \
    \"feature_store_type\": \"memory\", \
    \"serialize_protocol\": \"protobuf\", \
    \"inter_op_parallelism_threads\": 10, \
    \"intra_op_parallelism_threads\": 10, \
    \"init_timeout_minutes\": 1, \
    \"signature_name\": \"serving_default\", \
    \"read_thread_num\": 3, \
    \"update_thread_num\": 2, \
    \"model_store_type\": \"local\", \
    \"checkpoint_dir\": \"/home/deeprec/DeepRec/modelzoo/features/EmbeddingVariable/DeepFM/result/\", \
    \"savedmodel_dir\": \"/home/deeprec/DeepRec/modelzoo/features/EmbeddingVariable/DeepFM/savedmodels/1658806832/\" \
  } ";

struct input_format39array{
  float I1_13[13];
  char* C1_26[26];
};


::tensorflow::eas::ArrayProto get_proto_cc(void* char_input, int dim,::tensorflow::eas::ArrayDataType type){
  ::tensorflow::eas::ArrayShape array_shape;
  array_shape.add_dim(1);
  ::tensorflow::eas::ArrayProto input;
  input.set_dtype(type);


  switch(input.dtype()){
    case 1:
      input.add_float_val(*((float*)char_input));
      break;
    case 7:
      input.add_string_val((char*)char_input);
      break;
  }

  *(input.mutable_array_shape()) = array_shape;

  return input;
}




int main(int argc, char** argv) {
  
  // PLEASE EDIT THIS LINE!!!!
  char filepath[] = "/home/deeprec/DeepRec/modelzoo/features/EmbeddingVariable/WDL/test.csv";

  // // ------------------------------------------initialize serving model-----------------------------------------
  int state;
  void* model = initialize("", model_config, &state);
  if (state == -1) {
    std::cerr << "initialize error\n";
  }
   
  // // ---------------------------------------prepare serving data from file--------------------------------------
  
  FILE *fp = nullptr;
  char *line, *record;
  char buffer2[1024];
  char delim[] = ",";
  char end[] = "\n";
  int j = 0;
  int rows = 0;
  
  
  // get row number
  if ( (fp = fopen(filepath,"at+")) != nullptr) {
      while ((line = fgets(buffer2, sizeof(buffer2), fp)) != nullptr) rows++;
  }

  fclose(fp);
  
 
  // get rows
  char* all_elems[rows*39];
  int cur_pos = 0;

  if ( (fp = fopen(filepath,"at+")) != nullptr) {
      while ((line = fgets(buffer2, sizeof(buffer2), fp)) != NULL) {
          record = strtok(line, delim);
          while (record != NULL) {
              // only 1 label and 39 feature
              if (j >= 40) break;
              // disragard label 
              if (j == 0) {j++; record = strtok(NULL,delim); continue;}
              char* cur_item = (char*) malloc(sizeof(char)*strlen(record));
              strcpy(cur_item,record);
              if (cur_item[strlen(cur_item)-1] == *end) cur_item[strlen(cur_item)-1] = '\0';
              all_elems[cur_pos] = cur_item;

              cur_pos++;
              record = strtok(NULL, delim);
              j++;

          }
          j = 0;
         
      }
  }

  fclose(fp);

   

  // ----------------------------------------------prepare request input----------------------------------------------------
  for(int ii = 0; ii < rows; ii++ ){
        int start_idx = ii * 39;
        
        struct input_format39array inputs;
        for(int jj = 0; jj < 39; jj++){
          if( jj >= 0 && jj <= 12 ) inputs.I1_13[jj] = (float)(atof(all_elems[start_idx + jj]));
          else inputs.C1_26[jj-13] = (char*)all_elems[start_idx+jj];
        }

   
        // get input type
        ::tensorflow::eas::ArrayDataType dtype_f =
            ::tensorflow::eas::ArrayDataType::DT_FLOAT;

        ::tensorflow::eas::ArrayDataType dtype_s =
            ::tensorflow::eas::ArrayDataType::DT_STRING;

        // input setting
        ::tensorflow::eas::ArrayProto I1 =  get_proto_cc(&inputs.I1_13[0],1,dtype_f);
        ::tensorflow::eas::ArrayProto I2 =  get_proto_cc(&inputs.I1_13[1],1,dtype_f);
        ::tensorflow::eas::ArrayProto I3 =  get_proto_cc(&inputs.I1_13[2],1,dtype_f);
        ::tensorflow::eas::ArrayProto I4 =  get_proto_cc(&inputs.I1_13[3],1,dtype_f);
        ::tensorflow::eas::ArrayProto I5 =  get_proto_cc(&inputs.I1_13[4],1,dtype_f);
        ::tensorflow::eas::ArrayProto I6 =  get_proto_cc(&inputs.I1_13[5],1,dtype_f);
        ::tensorflow::eas::ArrayProto I7 =  get_proto_cc(&inputs.I1_13[6],1,dtype_f);
        ::tensorflow::eas::ArrayProto I8 =  get_proto_cc(&inputs.I1_13[7],1,dtype_f);
        ::tensorflow::eas::ArrayProto I9 =  get_proto_cc(&inputs.I1_13[8],1,dtype_f);
        ::tensorflow::eas::ArrayProto I10 = get_proto_cc(&inputs.I1_13[9],1,dtype_f);
        ::tensorflow::eas::ArrayProto I11 = get_proto_cc(&inputs.I1_13[10],1,dtype_f);
        ::tensorflow::eas::ArrayProto I12 = get_proto_cc(&inputs.I1_13[11],1,dtype_f);
        ::tensorflow::eas::ArrayProto I13 = get_proto_cc(&inputs.I1_13[12],1,dtype_f);
        ::tensorflow::eas::ArrayProto C1 =  get_proto_cc(inputs.C1_26[0],strlen(inputs.C1_26[0]),dtype_s);
        ::tensorflow::eas::ArrayProto C2 =  get_proto_cc(inputs.C1_26[1],strlen(inputs.C1_26[1]),dtype_s);
        ::tensorflow::eas::ArrayProto C3 =  get_proto_cc(inputs.C1_26[2],strlen(inputs.C1_26[2]),dtype_s);
        ::tensorflow::eas::ArrayProto C4 =  get_proto_cc(inputs.C1_26[3],strlen(inputs.C1_26[3]),dtype_s);
        ::tensorflow::eas::ArrayProto C5 =  get_proto_cc(inputs.C1_26[4],strlen(inputs.C1_26[4]),dtype_s);
        ::tensorflow::eas::ArrayProto C6 =  get_proto_cc(inputs.C1_26[5],strlen(inputs.C1_26[5]),dtype_s);
        ::tensorflow::eas::ArrayProto C7 =  get_proto_cc(inputs.C1_26[6],strlen(inputs.C1_26[6]),dtype_s);
        ::tensorflow::eas::ArrayProto C8 =  get_proto_cc(inputs.C1_26[7],strlen(inputs.C1_26[7]),dtype_s);
        ::tensorflow::eas::ArrayProto C9 =  get_proto_cc(inputs.C1_26[8],strlen(inputs.C1_26[8]),dtype_s);
        ::tensorflow::eas::ArrayProto C10 = get_proto_cc(inputs.C1_26[9],strlen(inputs.C1_26[9]),dtype_s);
        ::tensorflow::eas::ArrayProto C11 = get_proto_cc(inputs.C1_26[10],strlen(inputs.C1_26[10]),dtype_s);
        ::tensorflow::eas::ArrayProto C12 = get_proto_cc(inputs.C1_26[11],strlen(inputs.C1_26[11]),dtype_s);
        ::tensorflow::eas::ArrayProto C13 = get_proto_cc(inputs.C1_26[12],strlen(inputs.C1_26[12]),dtype_s);
        ::tensorflow::eas::ArrayProto C14 = get_proto_cc(inputs.C1_26[13],strlen(inputs.C1_26[13]),dtype_s);
        ::tensorflow::eas::ArrayProto C15 = get_proto_cc(inputs.C1_26[14],strlen(inputs.C1_26[14]),dtype_s);
        ::tensorflow::eas::ArrayProto C16 = get_proto_cc(inputs.C1_26[15],strlen(inputs.C1_26[15]),dtype_s);
        ::tensorflow::eas::ArrayProto C17 = get_proto_cc(inputs.C1_26[16],strlen(inputs.C1_26[16]),dtype_s);
        ::tensorflow::eas::ArrayProto C18 = get_proto_cc(inputs.C1_26[17],strlen(inputs.C1_26[17]),dtype_s);
        ::tensorflow::eas::ArrayProto C19 = get_proto_cc(inputs.C1_26[18],strlen(inputs.C1_26[18]),dtype_s);
        ::tensorflow::eas::ArrayProto C20 = get_proto_cc(inputs.C1_26[19],strlen(inputs.C1_26[19]),dtype_s);
        ::tensorflow::eas::ArrayProto C21 = get_proto_cc(inputs.C1_26[20],strlen(inputs.C1_26[20]),dtype_s);
        ::tensorflow::eas::ArrayProto C22 = get_proto_cc(inputs.C1_26[21],strlen(inputs.C1_26[21]),dtype_s);
        ::tensorflow::eas::ArrayProto C23 = get_proto_cc(inputs.C1_26[22],strlen(inputs.C1_26[22]),dtype_s);
        ::tensorflow::eas::ArrayProto C24 = get_proto_cc(inputs.C1_26[23],strlen(inputs.C1_26[23]),dtype_s);
        ::tensorflow::eas::ArrayProto C25 = get_proto_cc(inputs.C1_26[24],strlen(inputs.C1_26[24]),dtype_s);
        ::tensorflow::eas::ArrayProto C26 = get_proto_cc(inputs.C1_26[25],strlen(inputs.C1_26[25]),dtype_s);
        

        // PredictRequest
        ::tensorflow::eas::PredictRequest req;
        req.set_signature_name("serving_default");
        req.add_output_filter("Sigmoid:0");
      
        (*req.mutable_inputs())["I1:0"]  = I1;
        (*req.mutable_inputs())["I2:0"]  = I2;
        (*req.mutable_inputs())["I3:0"]  = I3;
        (*req.mutable_inputs())["I4:0"]  = I4;
        (*req.mutable_inputs())["I5:0"]  = I5;
        (*req.mutable_inputs())["I6:0"]  = I6;
        (*req.mutable_inputs())["I7:0"]  = I7;
        (*req.mutable_inputs())["I8:0"]  = I8;
        (*req.mutable_inputs())["I9:0"]  = I9;
        (*req.mutable_inputs())["I10:0"] = I10;
        (*req.mutable_inputs())["I11:0"] = I11;
        (*req.mutable_inputs())["I12:0"] = I12;
        (*req.mutable_inputs())["I13:0"] = I13;
        (*req.mutable_inputs())["C1:0"]  = C1;
        (*req.mutable_inputs())["C2:0"]  = C2;
        (*req.mutable_inputs())["C3:0"]  = C3;
        (*req.mutable_inputs())["C4:0"]  = C4;
        (*req.mutable_inputs())["C5:0"]  = C5;
        (*req.mutable_inputs())["C6:0"]  = C6;
        (*req.mutable_inputs())["C7:0"]  = C7;
        (*req.mutable_inputs())["C8:0"]  = C8;
        (*req.mutable_inputs())["C9:0"]  = C9;
        (*req.mutable_inputs())["C10:0"] = C10;
        (*req.mutable_inputs())["C11:0"] = C11;
        (*req.mutable_inputs())["C12:0"] = C12;
        (*req.mutable_inputs())["C13:0"] = C13;
        (*req.mutable_inputs())["C14:0"] = C14;
        (*req.mutable_inputs())["C15:0"] = C15;
        (*req.mutable_inputs())["C16:0"] = C16;
        (*req.mutable_inputs())["C17:0"] = C17;
        (*req.mutable_inputs())["C18:0"] = C18;
        (*req.mutable_inputs())["C19:0"] = C19;
        (*req.mutable_inputs())["C20:0"] = C20;
        (*req.mutable_inputs())["C21:0"] = C21;
        (*req.mutable_inputs())["C22:0"] = C22;
        (*req.mutable_inputs())["C23:0"] = C23;
        (*req.mutable_inputs())["C24:0"] = C24;
        (*req.mutable_inputs())["C25:0"] = C25;
        (*req.mutable_inputs())["C26:0"] = C26;
        

        size_t size = req.ByteSizeLong(); 
        void *buffer1 = malloc(size);
        req.SerializeToArray(buffer1, size);

        // ----------------------------------------------process and get feedback---------------------------------------------------
        void* output = nullptr;
        int output_size = 0;
        state = process(model, buffer1, size, &output, &output_size);

        // parse response
        std::string output_string((char*)output, output_size);
        ::tensorflow::eas::PredictResponse resp;
        resp.ParseFromString(output_string);
        std::cout << "process returned state: " << state << ", response: " << resp.DebugString();
  }

 //free memory
  for(int i=0; i < rows;i++){
      free(all_elems[i]);
  }

  return 0;
}


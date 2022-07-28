#include <iostream>
#include<stdlib.h>
#include "serving/processor/serving/processor.h"
#include "serving/processor/serving/predict.pb.h"
#include <vector>
#include <iostream>


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
    \"checkpoint_dir\": \"/home/deeprec/DeepRec/modelzoo/features/EmbeddingVariable/DIEN/result/\", \
    \"savedmodel_dir\": \"/home/deeprec/DeepRec/modelzoo/features/EmbeddingVariable/DIEN/savedmodels/1658740712/\" \
  } ";



::tensorflow::eas::ArrayProto get_proto_cc(std::vector<char*>& cur_vector, ::tensorflow::eas::ArrayDataType dtype_f){
  ::tensorflow::eas::ArrayShape array_shape;
  ::tensorflow::eas::ArrayProto input;
 
  int num_elem = (int)cur_vector.size();
  input.set_dtype(dtype_f);

  switch(dtype_f){
      case 1:
        array_shape.add_dim(1);
        if (num_elem == 1){
            input.add_float_val((float)atof(cur_vector.back()));
            *(input.mutable_array_shape()) = array_shape;
            return input;
        }
        array_shape.add_dim(cur_vector.size());
        for(unsigned int tt = 0; tt < cur_vector.size(); ++tt)
        {
          input.add_float_val((float)atof(cur_vector[tt]));
        }
        *(input.mutable_array_shape()) = array_shape;

        return input;
        
      break;

     case 3:
        array_shape.add_dim(1);
        if (num_elem == 1){
            input.add_int_val((int)atoi(cur_vector.back()));
            *(input.mutable_array_shape()) = array_shape;
            return input;
        }
        array_shape.add_dim(cur_vector.size());
        for(unsigned int tt = 0; tt < cur_vector.size(); ++tt)
        {
          input.add_int_val((int)atoi(cur_vector[tt]));
        }
        *(input.mutable_array_shape()) = array_shape;

        return input;
      break;
     
     default:
      break;
    }
    
    std::cerr << "type error\n";
    return input;
}




int main(int argc, char** argv) {

  char filepath[] = "/home/deeprec/DeepRec/modelzoo/features/EmbeddingVariable/DIEN/test_data.csv";

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
  char next_line[] = "k";
  int cur_type = 0;

  // vector variables
  std::vector<char*> cur_uids;
  std::vector<char*> cur_mids;
  std::vector<char*> cur_cats;
  std::vector<char*> cur_sl;        // single
  std::vector<char*> cur_mid_his;
  std::vector<char*> cur_cat_his;
  std::vector<char*> cur_mid_mask;
  std::vector<char*> cur_target;  // multiple

  // temp pointers
  std::vector<char*> temp_ptrs;
   
  // start read file
  if ( (fp = fopen(filepath,"at+")) != nullptr) {
      
      // read line by line
      while ((line = fgets(buffer2, sizeof(buffer2), fp)) != NULL) {
          
          // clear all vectors
          cur_uids.clear();
          cur_mids.clear();
          cur_cats.clear();
          cur_sl.clear();
          cur_mid_his.clear();
          cur_cat_his.clear();
          cur_mid_mask.clear();
          cur_target.clear();
          
          // reinitialize the type variable
          cur_type = 0;

          // free memory and clear ptrs
          for(int i = 0; i < (int)temp_ptrs.size(); ++i){free(temp_ptrs[i]);}
          temp_ptrs.clear();
          
          // traverse current line
          record = strtok(line, delim);
          while (record != nullptr) {
              // end of current line
              if (cur_type >= 8) {break;}
              // next type will start
              if (*record == *next_line) {cur_type++; record = strtok(NULL,delim); continue;}
              // switch
              switch (cur_type)
              {

                case 0:

                  temp_ptrs.push_back((char*) malloc(sizeof(char)*strlen(record)));
                  strcpy(temp_ptrs.back(),record);
                  cur_uids.push_back(temp_ptrs.back());
                  break;

                case 1:

                  temp_ptrs.push_back((char*) malloc(sizeof(char)*strlen(record)));
                  strcpy(temp_ptrs.back(),record);
                  cur_mids.push_back(temp_ptrs.back());
                  break;

                case 2:

                  temp_ptrs.push_back((char*) malloc(sizeof(char)*strlen(record)));
                  strcpy(temp_ptrs.back(),record);
                  cur_cats.push_back(temp_ptrs.back());
                  break;

                case 3:

                  temp_ptrs.push_back((char*) malloc(sizeof(char)*strlen(record)));
                  strcpy(temp_ptrs.back(),record);
                  cur_mid_his.push_back(temp_ptrs.back());
                  break;

                case 4:

                  temp_ptrs.push_back((char*) malloc(sizeof(char)*strlen(record)));
                  strcpy(temp_ptrs.back(),record);
                  cur_cat_his.push_back(temp_ptrs.back());
                  break;

                case 5:

                  temp_ptrs.push_back((char*) malloc(sizeof(char)*strlen(record)));
                  strcpy(temp_ptrs.back(),record);
                  cur_mid_mask.push_back(temp_ptrs.back());
                  break;

                case 6:

                  temp_ptrs.push_back((char*) malloc(sizeof(char)*strlen(record)));
                  strcpy(temp_ptrs.back(),record);
                  cur_target.push_back(temp_ptrs.back());
                  break;

                case 7:

                  temp_ptrs.push_back((char*) malloc(sizeof(char)*strlen(record)));
                  strcpy(temp_ptrs.back(),record);
                  cur_sl.push_back(temp_ptrs.back());
                  break;

                default:
                  break;
              }  

              record = strtok(NULL,delim);

          } 
         

          ::tensorflow::eas::ArrayDataType dtype_i =
            ::tensorflow::eas::ArrayDataType::DT_INT32;
          ::tensorflow::eas::ArrayDataType dtype_f =
            ::tensorflow::eas::ArrayDataType::DT_FLOAT;
          // get all inputs
          ::tensorflow::eas::ArrayProto proto_uids    = get_proto_cc(cur_uids,dtype_i);               // -1
          ::tensorflow::eas::ArrayProto proto_mids    = get_proto_cc(cur_mids,dtype_i);               // -1
          ::tensorflow::eas::ArrayProto proto_cats    = get_proto_cc(cur_cats,dtype_i);               // -1
          ::tensorflow::eas::ArrayProto proto_mid_his = get_proto_cc(cur_mid_his,dtype_i);            // -1 -1
          ::tensorflow::eas::ArrayProto proto_cat_his = get_proto_cc(cur_cat_his,dtype_i);            // -1 -1
          ::tensorflow::eas::ArrayProto proto_mid_mask= get_proto_cc(cur_mid_mask,dtype_f); //float // -1 -1
          ::tensorflow::eas::ArrayProto proto_target  = get_proto_cc(cur_target,dtype_f); //float   // -1 -1
          ::tensorflow::eas::ArrayProto proto_sl      = get_proto_cc(cur_sl,dtype_i);                 // -1


          // setup request
          ::tensorflow::eas::PredictRequest req;
          req.set_signature_name("serving_default");
          req.add_output_filter("top_full_connect/add_2:0");
        
          (*req.mutable_inputs())["Inputs/uid_batch_ph:0"]     = proto_uids;
          (*req.mutable_inputs())["Inputs/mid_batch_ph:0"]     = proto_mids;
          (*req.mutable_inputs())["Inputs/cat_batch_ph:0"]     = proto_cats;
          (*req.mutable_inputs())["Inputs/mid_his_batch_ph:0"] = proto_mid_his;
          (*req.mutable_inputs())["Inputs/cat_his_batch_ph:0"] = proto_cat_his;
          (*req.mutable_inputs())["Inputs/mask:0"]         = proto_mid_mask;
          (*req.mutable_inputs())["Inputs/target_ph:0"]        = proto_target;
          (*req.mutable_inputs())["Inputs/seq_len_ph:0"]       = proto_sl;

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
  }

  fclose(fp);
  return 0;
}


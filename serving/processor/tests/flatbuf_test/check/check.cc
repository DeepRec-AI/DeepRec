#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include "request_generated.h"

std::string ReadFileIntoString(
    const std::string& filename) {
  std::ifstream ifile(filename);
  std::ostringstream buf;
  char ch; 
  while(buf && ifile.get(ch)) {
    buf.put(ch);
  }

  return buf.str();
}

int main() {
  // read a bin file which generate by
  // other lang like python and java
  // file path: /tmp/request_buf.bin
  std::string req_str = ReadFileIntoString("/tmp/request_buf.bin");
  const eas::PredictRequest* flat_recv_req =
      flatbuffers::GetRoot<eas::PredictRequest>((void*)(req_str.data()));

  std::cout << "signature_name = " << flat_recv_req->signature_name()->str() << "\n";
  int idx = 0;
  for (auto fname : *(flat_recv_req->feed_names())) {
    std::cout << "#" << idx++ << "_feed_name = " << fname->str() << "\n";
  }   
  idx = 0;
  for (auto t : *(flat_recv_req->types())) {
    std::cout << "#" << idx++ << "_type = " << t << "\n";
  }   
  idx = 0;
  for (auto s : *(flat_recv_req->shapes())) {
    std::cout << "#" << idx++ << "_dim = ";
    for (auto d : *(s->dim())) {
      std::cout << d << " ";
    }   
    std::cout << "\n";
  }   
  idx = 0;
  for (auto fname : *(flat_recv_req->fetch_names())) {
    std::cout << "#" << idx++ << "_fetch_name = " << fname->str() << "\n";
  }
  std::cout << "float content: \n"; 
  idx = 0;
  for (auto c : *(flat_recv_req->float_content())) {
    for (auto num : *(c->content())) {
      std::cout << "#" << idx++ << "_num, num = " << num << "\n";
    }
    std::cout << "\n";
  }
  std::cout << "int64 content: \n";
  idx = 0;
  for (auto c : *(flat_recv_req->i64_content())) {
    for (auto num : *(c->content())) {
      std::cout << "#" << idx++ << "_num, num = " << num << "\n";
    }
    std::cout << "\n";
  }
  std::cout << "double content: \n";
  idx = 0;
  for (auto c : *(flat_recv_req->d_content())) {
    for (auto num : *(c->content())) {
      std::cout << "#" << idx++ << "_num, num = " << num << "\n";
    }
    std::cout << "\n";
  }
  std::cout << "int content: \n";
  idx = 0;
  for (auto c : *(flat_recv_req->i_content())) {
    for (auto num : *(c->content())) {
      std::cout << "#" << idx++ << "_num, num = " << num << "\n";
    }
    std::cout << "\n";
  }
  std::cout << "string content len: \n";
  idx = 0;
  std::vector<int> string_content_len_array;
  for (auto len : *(flat_recv_req->string_content_len())) {
    std::cout << "#" << idx++ << "_string_content_len = " << len << "\n";
    string_content_len_array.emplace_back(len);
  }
  std::cout << "string content: \n";
  idx = 0;
  int cur = 0;
  for (auto c : *(flat_recv_req->string_content())) {
    int str_count = string_content_len_array[cur++];
    int offset = 0;
    int64_t total_len = 0;
    for (int x = 0; x < str_count; ++x) {
      total_len += string_content_len_array[cur];
      std::cout << "#" << idx++ << "_string_content = "
                << std::string(c->c_str()+offset, string_content_len_array[cur]) << "\n";
      offset += string_content_len_array[cur++];
    }
    assert(total_len == c->size());
  }

  return 0;
}


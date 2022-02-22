#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_FUSED_EMBEDDING_CU_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_FUSED_EMBEDDING_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"

#define CK_CUDA_THROW_(x)                                                      \
  do {                                                                         \
    cudaError_t retval = (x);                                                  \
    if (retval != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("Runtime error: ") +                \
                               (cudaGetErrorString(retval)) + " " + __FILE__ + \
                               ":" + std::to_string(__LINE__) + " \n");        \
    }                                                                          \
  } while (0)

namespace tensorflow {

namespace fused_embedding {

template <typename T>
inline T* data_p_with_type(Tensor& t) {
  return reinterpret_cast<T*>(t.data());
}

template <typename T>
inline T* data_p_with_type(const Tensor& t) {
  return reinterpret_cast<T*>(t.data());
}

template <typename T>
inline T* data_p_with_type(Tensor* t) {
  return reinterpret_cast<T*>(t->data());
}

template <typename T>
inline T* data_p_with_type(const Tensor* t) {
  return reinterpret_cast<T*>(t->data());
}

struct IndicePair {
  int64_t row_in_batch;
  int64_t entry_in_column;
};

}  // namespace fused_embedding

}  // namespace tensorflow


// =================================================================================================
#include <cuda_runtime.h>
#include <functional>
#include <numeric>
#include <stdio.h>
#include <string>
#include <vector>
template<typename T>
void printGPUTensorHelper(const T* src,
                          const std::vector<int>& dims,
                          const std::vector<int>& max_num_to_print,
                          bool if_exit = false)
{
    int element_num = 1;
    int dim_num = (int)dims.size();

    printf("Tensor size: (");
    for (int i = 0; i < dim_num; i++) {
        element_num *= dims[i];
        printf("%d, ", dims[i]);
    }
    printf(")\n");
    T* host_buffer = new T[element_num];
    cudaMemcpy((void*)host_buffer, (void*)src, sizeof(T) * element_num, cudaMemcpyDeviceToHost);

    std::vector<int> strides(dim_num, 1);
    for (int i = dim_num - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
        printf("%d ", strides[i]);
    }

    cudaDeviceSynchronize();

    printf("================== start printing tensor =====================\n");

    std::function<void(int, int)> recursive_print;
    recursive_print =
        [&recursive_print, host_buffer, &dims = dims, &max_num_to_print = max_num_to_print, &strides = strides](
            int current_dim, int current_dim_offset) -> void {
        if (current_dim < dims.size() - 1) {
            for (int i = 0; i < current_dim; i++) {
                printf(" ");
            }
            printf("[ dim %d\n", current_dim);

            if (2 * max_num_to_print[current_dim] >= dims[current_dim]) {
                for (int i = 0; i < dims[current_dim]; i++) {
                    recursive_print(current_dim + 1, current_dim_offset + i * strides[current_dim]);
                }
            }
            else {
                for (int i = 0; i < max_num_to_print[current_dim]; i++) {
                    recursive_print(current_dim + 1, current_dim_offset + i * strides[current_dim]);
                }
                for (int i = 0; i < current_dim; i++) {
                    printf(" ");
                }
                printf(" ......\n");
                for (int i = dims[current_dim] - max_num_to_print[current_dim]; i < dims[current_dim]; i++) {
                    recursive_print(current_dim + 1, current_dim_offset + i * strides[current_dim]);
                }
            }
            for (int i = 0; i < current_dim; i++) {
                printf(" ");
            }
            printf("]\n");
        }
        else {
            for (int i = 0; i < current_dim; i++) {
                printf(" ");
            }
            printf("[ ");
            printf("dim %d, offset: %d, ", current_dim, current_dim_offset);
            if (2 * max_num_to_print[current_dim] >= dims[current_dim]) {
                for (int i = 0; i < dims[current_dim]; i++) {
                    printf("%s ", std::to_string(host_buffer[current_dim_offset + i * strides[current_dim]]).c_str());
                }
            }
            else {
                for (int i = 0; i < max_num_to_print[current_dim]; i++) {
                    printf("%s ", std::to_string(host_buffer[current_dim_offset + i * strides[current_dim]]).c_str());
                }
                printf("...... ");
                for (int i = dims[current_dim] - max_num_to_print[current_dim]; i < dims[current_dim]; i++) {
                    printf("%s ", std::to_string(host_buffer[current_dim_offset + i * strides[current_dim]]).c_str());
                }
            }
            printf("]\n");
        }
    };

    recursive_print(0, 0);

    printf("\n================== stop =====================\n");
    delete[] host_buffer;

    if (if_exit) {
        exit(0);
    }
};

#include <fstream>
#include <stdio.h>
template<typename T>
void dumpGPUTensorHelper(const T* src, const std::vector<int>& dims, const std::string& out_file, bool if_exit = true)
{
    int element_num = 1;
    int dim_num = (int)dims.size();

    printf("Tensor size: (");
    for (int i = 0; i < dim_num; i++) {
        element_num *= dims[i];
        printf("%d, ", dims[i]);
    }
    printf(")\n");
    T* host_buffer = new T[element_num];
    cudaMemcpy((void*)host_buffer, (void*)src, sizeof(T) * element_num, cudaMemcpyDeviceToHost);

    std::ofstream file(out_file, std::ios::binary);
    if (file.is_open()) {
        file.write((const char*)host_buffer, sizeof(T) * element_num);
        file.close();
        std::cout << "Dumped tensor to " << out_file << std::endl;
    }
    else {
        std::cout << "Unable to open " << out_file << std::endl;
    }

    if (if_exit) {
        exit(0);
    }
}


#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_FUSED_EMBEDDING_CU_H_
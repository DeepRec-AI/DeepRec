/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "facade.h"
#include "parameters/raw_manager.h"
#include "tensor_buffer/tf_tensor_wrapper.h"
#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif
#include <stdexcept>
#include <atomic>
#include <regex>

namespace SparseOperationKit {

Facade::Facade() 
: resources_mgr_(ResourcesManager::Create()),
params_mgr_(RawManager::Create(resources_mgr_)),
embedding_mgr_(EmbeddingManager::Create(resources_mgr_))
{
}


Facade* Facade::instance() {
    static Facade instance;
    return &instance;
}

void Facade::operator delete(void*) {
    throw std::domain_error("This pointer cannot be manually deleted.");
}


void Facade::get_nccl_unique_id(std::string& nccl_unique_id) {
    ResourcesManager::get_nccl_unique_id(nccl_unique_id);
}

void Facade::get_nccl_unique_id(int32_t* nccl_unique_id) {
    ResourcesManager::get_nccl_unique_id(nccl_unique_id);
}

void Facade::get_random_seed(uint64_t* seed) {
    ResourcesManager::get_random_seed(seed);
}

void Facade::init(const size_t global_replica_id, const size_t num_replicas_in_sync, 
                  const int32_t* nccl_unique_id, const uint64_t global_seed,
                  const size_t global_batch_size, const cudaStream_t& tf_stream) {
    // initialize resource manager
    resources_mgr_->init(global_replica_id, num_replicas_in_sync, nccl_unique_id, global_seed, tf_stream);
    // initialize parameters manager
    params_mgr_->init(global_replica_id);
    // initialize embedding manager
    embedding_mgr_->init(global_replica_id, global_batch_size);
    // initialize other parts TODO:

    auto create_mutexs = [this]() {
        init_mus_.clear();
        const size_t local_gpu_count = resources_mgr_->get_local_gpu_count();
        init_mus_.insert(init_mus_.begin(), local_gpu_count, nullptr);
        for (size_t i = 0; i < local_gpu_count; i++) {
            init_mus_[i] = std::make_shared<std::mutex>();
        }
    };
    resources_mgr_->blocking_call_once(create_mutexs);
}

void Facade::generate_unique_name(const bool trainable, std::string &variable_name) {
    params_mgr_->gen_unique_name(trainable, variable_name);
}

void Facade::create_variables(const size_t local_replica_id, const float* initial_value, const bool use_hashtable,
                              const std::vector<int64_t> shape, const std::string name,
                              const bool trainable, 
                              tensorflow::core::RefCountPtr<tensorflow::EmbeddingVariable>& emb_variable,
                              tensorflow::Tensor* emb_tensor) {
    throw std::runtime_error("Not implemented yet.");
}

/*This function will be called multiple times sequentially*/
void Facade::create_variables(const size_t local_replica_id, const std::string& initializer, const bool use_hashtable,
                              const std::vector<int64_t> shape, const std::string name,
                              const bool trainable, 
                              tensorflow::core::RefCountPtr<tensorflow::EmbeddingVariable>& emb_variable,
                              tensorflow::Tensor* emb_tensor) {
    try {
        std::shared_ptr<ParamInterface> param;
        std::vector<size_t> _shape(shape.size());
        for (size_t i = 0; i < shape.size(); i++) _shape[i] = static_cast<size_t>(shape[i]);
        params_mgr_->create_variables(initializer, use_hashtable, _shape, name, trainable, param);

        auto emb_buffer_builder = EmbeddingBufferBuilder::create(param->get_embedding_table_tensor(local_replica_id));
        auto buffer = emb_buffer_builder->get_init_buffer();
        // FIXME: in TF 2.4, int64_t is not equal to long long
        const std::vector<tensorflow::int64> temp_shape(shape.begin(), shape.end());
        tensorflow::TensorShape tensor_shape = tensorflow::TensorShape(temp_shape);
        *emb_tensor = tensorflow::Tensor(/*type=*/tensorflow::DT_FLOAT,
                                        /*shape=*/tensor_shape,
                                        /*buf=*/buffer);
        params_mgr_->push_back_embedding_buffer_builder(local_replica_id, emb_buffer_builder);

        emb_variable->set_param(param);

    } catch (const std::exception& error) {
        throw std::runtime_error(ErrorBase + error.what());
    }
}

void Facade::create_variables(const size_t local_replica_id, float* variable, const bool use_hashtable,
                              const std::vector<int64_t> shape, const std::string name,
                              const bool trainable, 
                              tensorflow::core::RefCountPtr<tensorflow::EmbeddingVariable>& emb_variable,
                              tensorflow::Tensor* emb_tensor) {
    throw std::runtime_error("Not implemented yet.");
}

void Facade::create_embedding_sparse(const tensorflow::core::RefCountPtr<tensorflow::EmbeddingVariable>& variable,
                                     const std::string input_dispatcher,
                                     const std::vector<std::string>& input_dispatcher_subsequent_ops,
                                     const std::string embedding_executor,
                                     const std::string output_dispatcher,
                                     const std::vector<std::string>& output_dispatcher_subsequent_ops,
                                     const size_t slot_num, const size_t max_nnz, const size_t max_feature_num, 
                                     const std::string combiner, 
                                     tensorflow::Tensor* emb_handle) {
    // check input validness
    CombinerType combiner_type_enum;
    find_item_in_map(CombinerMap, combiner, combiner_type_enum);
    if (slot_num <= 0) throw std::runtime_error(ErrorBase + "slot_num must be >= 1.");
    if (max_nnz <= 0) throw std::runtime_error(ErrorBase + "max_nnz must be >= 1.");
    if (!(max_feature_num >= 1 && max_feature_num <= slot_num * max_nnz))
        throw std::runtime_error(ErrorBase + "max_feature_num must be in range of [1, slot_num * max_nnz]");

    // get variable pointer
    std::shared_ptr<ParamInterface> param;
    variable->get_param(param);

    // create embedding layer based on those components
    std::shared_ptr<EmbeddingLayer> embedding;
    embedding_mgr_->create_embedding(param, 
                                     input_dispatcher, input_dispatcher_subsequent_ops,
                                     embedding_executor,
                                     output_dispatcher, output_dispatcher_subsequent_ops,
                                     slot_num, max_nnz, max_feature_num, combiner_type_enum,
                                     embedding);

    // set output
    StoreEmbeddingInVariantTensor(embedding, emb_handle);
}

void Facade::create_embedding_dense(const tensorflow::core::RefCountPtr<tensorflow::EmbeddingVariable>& variable,
                                    const std::string input_dispatcher,
                                    const std::vector<std::string>& input_dispatcher_subsequent_ops,
                                    const std::string embedding_executor,
                                    const std::string output_dispatcher,
                                    const std::vector<std::string>& output_dispatcher_subsequent_ops,
                                    const size_t slot_num, const size_t nnz_per_slot,
                                    tensorflow::Tensor* emb_handle) {
    // check input validness
    if (slot_num <= 0) throw std::runtime_error(ErrorBase + "slot_num must be >= 1.");
    if (nnz_per_slot <= 0) throw std::runtime_error(ErrorBase + "nnz_per_slot must be >= 1.");

    // create embedding layer
    std::shared_ptr<ParamInterface> param;
    variable->get_param(param);

    std::shared_ptr<EmbeddingLayer> embedding;
    embedding_mgr_->create_embedding(param, input_dispatcher, input_dispatcher_subsequent_ops,
                                     embedding_executor, 
                                     output_dispatcher, output_dispatcher_subsequent_ops,
                                     slot_num, nnz_per_slot,
                                     embedding);

    // store output
    StoreEmbeddingInVariantTensor(embedding, emb_handle);
}

void Facade::create_optimizer(const std::string optimizer_type,
                              tensorflow::Tensor* optimizer_handle,
                              optimizer_hyper_params hyper_params) {
    static bool optimizer_created = false;
    if (optimizer_created) throw std::runtime_error(ErrorBase + "There already exists an optimizer instance. " +
                                                    "Do not call create_custom_optimizer() more than once.");
    OptimizerType optimizer_type_enum;
    find_item_in_map(OptimizerMap, optimizer_type, optimizer_type_enum);

    // generate an optimizer instance
    optimizer_ = Optimizer::Get(optimizer_type_enum, std::move(hyper_params), 
                                params_mgr_, resources_mgr_);
    optimizer_created = true;

    // set output
    StoreOptimizerInVariantTensor(optimizer_, optimizer_handle);
}

/*This function will be called by multiple threads simultaneously*/
void Facade::try_allocate_memory(const size_t global_replica_id) const {
    static std::atomic<bool> allocated_{false};

    auto allocate_memory_helper = [this](const size_t global_replica_id) {
        if (optimizer_) {
            auto optimizer_helper = [this](){
                optimizer_->create_preparers(embedding_mgr_);
                optimizer_->reserve_spaces();
            };
            resources_mgr_->blocking_call_once(optimizer_helper);
        } // if optimizer_ is valid

        resources_mgr_->allocate_memory(global_replica_id);
        params_mgr_->allocate_memory(global_replica_id);
        embedding_mgr_->allocate_memory(global_replica_id);
    };

    auto set_allocated_helper = [this](const size_t global_replica_id) mutable {
        auto _Func = []() mutable {
            allocated_.store(true, std::memory_order_release);
            MESSAGE("SparseOperationKit allocated internal memory.");
        };
        resources_mgr_->blocking_call_once(_Func);
    };

    bool allocated = allocated_.load(std::memory_order_acquire);
    if (!allocated) { // first check
        const size_t local_replica_id = resources_mgr_->cal_local_id_from_global_id(global_replica_id);
        std::lock_guard<std::mutex> lock(*(init_mus_[local_replica_id]));
        allocated = allocated_.load(std::memory_order_relaxed);
        if (!allocated) { // second check
            allocate_memory_helper(global_replica_id);
            set_allocated_helper(global_replica_id);
        }
    } 
    return;
}

/*This function will only be called by one CPU threads.*/
void Facade::try_allocate_memory() {
    static std::atomic<bool> allocated{false};
    if (allocated.load()) return;

    std::lock_guard<std::mutex> lock(mu_);
    // check again to see if another thread has allocated memory
    if (allocated.load()) return;

    auto try_allocate_helper = [this](size_t local_device_id) {
        HugeCTR::CudaDeviceContext context;
        context.set_device(resources_mgr_->get_local_gpu(local_device_id)->get_local_device_id());
        const size_t global_replica_id = resources_mgr_->cal_global_id_from_local_id(local_device_id);
        try_allocate_memory(global_replica_id);
    };
    for (size_t id = 0; id < resources_mgr_->get_local_gpu_count(); id++) 
        resources_mgr_->push_to_threadpool(try_allocate_helper, id);
    resources_mgr_->sync_threadpool();
    allocated.store(true);
}

void Facade::get_output_shape(const tensorflow::Tensor* emb_handle,
                              tensorflow::TensorShape& tensor_shape,
                              const bool dynamic_input) {
    std::shared_ptr<EmbeddingLayer> embedding;
    GetEmbeddingFromVariantTensor(emb_handle, embedding);

    std::vector<int64_t> output_shape;
    embedding_mgr_->get_output_shape(embedding, output_shape, dynamic_input);

    // FIXME: in TF 2.4, int64_t is not equal to long long int.
    const std::vector<tensorflow::int64> _output_shape(output_shape.begin(), output_shape.end());
    tensor_shape = std::move(tensorflow::TensorShape(_output_shape));
}

void Facade::get_grad_shape(const size_t global_replica_id,
                            const tensorflow::Tensor* emb_handle,
                            tensorflow::TensorShape& grad_shape) {
    std::shared_ptr<EmbeddingLayer> embedding;
    GetEmbeddingFromVariantTensor(emb_handle, embedding);

    std::vector<int64_t> _grad_shape;
    embedding_mgr_->get_grad_shape(global_replica_id, embedding, _grad_shape);

    // FIXME: in TF 2.4, int64_t is not equal to long long int.
    const std::vector<tensorflow::int64> temp_grad_shape(_grad_shape.begin(), _grad_shape.end());
    grad_shape = std::move(tensorflow::TensorShape(temp_grad_shape));
}

void Facade::forward(const tensorflow::Tensor* emb_handle, 
                     const tensorflow::Tensor* values_tensor,
                     const tensorflow::Tensor* indices_tensor,
                     const size_t global_replica_id,
                     const bool training,
                     tensorflow::Tensor* emb_vector_tensor) {
#ifdef USE_NVTX
    nvtxRangeId_t forward_marker = nvtxRangeStartA("forward");
#endif
    // try to allocate internal memory
    try_allocate_memory(global_replica_id);

    std::shared_ptr<EmbeddingLayer> embedding;
    GetEmbeddingFromVariantTensor(emb_handle, embedding);

    const std::shared_ptr<Tensor> values = TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(values_tensor));
    const std::shared_ptr<Tensor> indices = TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(indices_tensor));
    std::shared_ptr<Tensor> emb_vector = TFTensorWrapper::create(emb_vector_tensor);

    // delegate embedding forward to embedding manager
    embedding_mgr_->forward(embedding, values, indices, global_replica_id, training, emb_vector);
#ifdef USE_NVTX
    nvtxRangeEnd(forward_marker);
#endif
}

void Facade::forward(const tensorflow::Tensor* emb_handle, 
                     const tensorflow::Tensor* values_tensor,
                     const size_t global_replica_id,
                     const bool training,
                     tensorflow::Tensor* emb_vector_tensor) {
#ifdef USE_NVTX
    nvtxRangeId_t forward_marker = nvtxRangeStartA("forward");
#endif
    // try to allocate internal memory
    try_allocate_memory(global_replica_id);

    std::shared_ptr<EmbeddingLayer> embedding;
    GetEmbeddingFromVariantTensor(emb_handle, embedding);

    const std::shared_ptr<Tensor> values = TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(values_tensor));
    std::shared_ptr<Tensor> emb_vector = TFTensorWrapper::create(emb_vector_tensor);

    // delegate embedding forward to embedding manager
    embedding_mgr_->forward(embedding, values, global_replica_id, training, emb_vector);
#ifdef USE_NVTX
    nvtxRangeEnd(forward_marker);
#endif
}

void Facade::backward(const tensorflow::Tensor* emb_handle,
                      const size_t global_replica_id,
                      const tensorflow::Tensor* top_gradient_tensor,
                      tensorflow::Tensor* gradient_tensor,
                      tensorflow::Tensor* value_index_tensor) {
#ifdef USE_NVTX
    nvtxRangeId_t backward_marker = nvtxRangeStartA("backward");
#endif
    std::shared_ptr<EmbeddingLayer> embedding;
    GetEmbeddingFromVariantTensor(emb_handle, embedding);

    const std::shared_ptr<Tensor> top_gradient = TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(top_gradient_tensor));
    std::shared_ptr<Tensor> gradient = TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(gradient_tensor));
    std::shared_ptr<Tensor> value_index = TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(value_index_tensor));

    // delegate embedding backward to embedding manager
    embedding_mgr_->backward(embedding, top_gradient, global_replica_id, gradient, value_index);

#ifdef USE_NVTX
    nvtxRangeEnd(backward_marker);
#endif
}

void Facade::apply_gradients(const tensorflow::core::RefCountPtr<tensorflow::EmbeddingVariable>& variable,
                            const tensorflow::Tensor* gradient_tensor,
                            const tensorflow::Tensor* local_indices_tensor,
                            const size_t local_replica_id,
                            const float learning_rate,
                            const size_t current_step) {
#ifdef USE_NVTX
    nvtxRangeId_t apply_grad_marker = nvtxRangeStartA("apply_gradients");
#endif
    // get ParamInterface from variable
    std::shared_ptr<ParamInterface> param;
    variable->get_param(param);

    // apply gradients
    auto grad = TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(gradient_tensor));
    auto local_indices = TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(local_indices_tensor));

    optimizer_->apply_gradients(param, grad, local_indices, local_replica_id, learning_rate, current_step);

#ifdef USE_NVTX
    nvtxRangeEnd(apply_grad_marker);
#endif
}


void Facade::dump_to_file(const tensorflow::core::RefCountPtr<tensorflow::EmbeddingVariable>& emb_variable,
                          const std::string filepath) {
    // get param handle
    std::shared_ptr<ParamInterface> param;
    emb_variable->get_param(param);

    // delegate dump to file to param manager.
    params_mgr_->dump_to_file(param, filepath);
}

void Facade::restore_from_file(tensorflow::core::RefCountPtr<tensorflow::EmbeddingVariable>& emb_variable,
                               const std::string filepath) {
    // try to allocate internal memory
    try_allocate_memory();

    // get param handle
    std::shared_ptr<ParamInterface> param;
    emb_variable->get_param(param);

    // delegate restore from file to param manager.
    params_mgr_->restore_from_file(param, filepath);

}

void Facade::load_embedding_values(tensorflow::core::RefCountPtr<tensorflow::EmbeddingVariable>& emb_variable,
                                 const tensorflow::OpInputList* tensor_list) {
    if (tensor_list->size() < 1) throw std::runtime_error(ErrorBase + "There must be at least one tensor.");

    // try to allocate internal memory
    try_allocate_memory();

    // get param handle
    std::shared_ptr<ParamInterface> param;
    emb_variable->get_param(param);

    std::vector<std::shared_ptr<Tensor>> tensors;
    const size_t emb_vec_size = param->get_embedding_vec_size();
    for (auto iter = tensor_list->begin(); iter != tensor_list->end(); iter++) {
        const tensorflow::Tensor* tensor = iter.operator->();

        if (!tensorflow::TensorShapeUtils::IsMatrix(tensor->shape()))
            throw std::runtime_error(ErrorBase + 
                "The shape of tensor must be [sub_vocabulary_size, embedding_vec_size].");
        if (static_cast<size_t>(tensor->dim_size(1)) != emb_vec_size)
            throw std::runtime_error(ErrorBase + "embedding_vec_size dimension is not consistent.");
        
        tensors.emplace_back(TFTensorWrapper::create(const_cast<tensorflow::Tensor*>(tensor)));
    } // iter in tensor_list

    // delegate this job to param manager
    params_mgr_->load_embedding_values(param, tensors);
}

// backdoors for unit test
const std::shared_ptr<ResourcesManager>& Facade::get_resource_mgr() const {
    return resources_mgr_;
}


int32_t GetLocalReplicaIdFromDeviceName(const std::string device_name) {
    try {
        auto find_str = [](const std::string input, const char* pattern) {
            std::regex reg(pattern);
            std::smatch result;
            if (std::regex_search(input, result, reg)) 
                return std::string(result.str());
            else throw std::runtime_error("Cannot find " + std::string(pattern) + 
                                          " in " + input);
        };
        constexpr char pattern1[] = "/device:GPU:\\d+$";
        constexpr char pattern2[] = "\\d+$";
        const std::string num = find_str(find_str(device_name, pattern1), pattern2);
        return string2num(num);
    } catch (const std::exception& error) {
        throw std::runtime_error(ErrorBase + error.what());
    }
}


} // namespace SparseOperationKit
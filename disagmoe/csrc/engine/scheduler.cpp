#include "scheduler.h"
#include "utils.hpp"

#include <exception>

Scheduler_t Scheduler::build(MuPool_t pool, std::vector<int> layer_ids, std::string policy = "largest") {
    if (policy == "largest") {
        return std::make_shared<Scheduler>(LargestScheduler(pool, layer_ids));
    } else {
        throw std::runtime_error(policy + " schedule not implemented.");
    }
}

LargestScheduler::LargestScheduler(MuPool_t pool, std::vector<int> layer_ids):
    Scheduler(pool, layer_ids, "largest") {

    }

Scheduler::Scheduler(MuPool_t pool, std::vector<int> layer_ids, std::string policy = "largest"): 
    pool(pool), layer_ids(layer_ids), policy(policy) {
    
}

TensorBatch Scheduler::merge(std::vector<TensorBatch> batches) {
    std::vector<metadata_t> metas(batches.size());
    for (size_t i = 0; i < batches.size(); i ++)
        metas[i] = batches[i].metadata;
    auto meta = Metadata::concat(metas);

    auto dtype = meta->get_datatype_size();
    
    uintptr_t buf = alloc_cuda_tensor(
        meta->num_element() * dtype, 
        this->pool->get_device_id()
    );
    
    uintptr_t ptr = buf;
    for (auto batch: batches) {
        auto size = batch.metadata->num_element() * dtype;
        cudaMemcpy((void*) ptr, (void*) batch.data, size, 
            cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        ptr += size;
    }

    return TensorBatch {ptr, meta};
}

TensorBatch Scheduler::schedule() {
    auto batches = this->_schedule();
    auto batch = this->merge(batches);
    return batch;
}

void Scheduler::wait_for_new_requests() {
    pool->wait_for_new_requests();
}

std::vector<TensorBatch> LargestScheduler::_schedule() {
    return pool->fetch_largest_batch();
}
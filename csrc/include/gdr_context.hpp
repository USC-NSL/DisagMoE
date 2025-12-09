#pragma once

#include <gdrapi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <torch/torch.h>
#include "cuda_utils.h"

#include <iostream>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

/**
 * @brief RAII wrapper around a GPU memory region registered with GDRCopy.
 *
 * This context allows direct CPU writes into GPU memory (BAR1 mapping)
 * without going through cudaMemcpy. Ideal for small or latency-sensitive transfers.
 */
class GdrContext {
public:
    // ============================================================
    // Constructors
    // ============================================================

    /// Construct directly from a torch::Tensor
    explicit GdrContext(const torch::Tensor& tensor)
        : tensor_(tensor), mh_{}, bar_ptr_{nullptr}, cpu_ptr_{nullptr}, 
          size_{tensor.nbytes()}, dev_ptr_{reinterpret_cast<uintptr_t>(tensor.data_ptr())} {

        if (!tensor.is_cuda())
            throw std::runtime_error("GdrContext: tensor must be CUDA tensor");
        if (!tensor.is_contiguous())
            throw std::runtime_error("GdrContext: tensor must be contiguous");

        initialize(reinterpret_cast<uint64_t>(tensor.data_ptr()), tensor.nbytes());
    }

    GdrContext() = default;

    /// Destructor: automatically unmaps and unpins
    ~GdrContext() noexcept { cleanup(); }

    // Non-copyable
    GdrContext(const GdrContext&) = delete;
    GdrContext& operator=(const GdrContext&) = delete;

    // ============================================================
    // Core APIs
    // ============================================================

    inline torch::Tensor get_tensor() const { return tensor_; }

    /// Copy from host memory into GPU buffer via BAR1
    inline void copy_from_host(const void* src, size_t nbytes, size_t dst_offset = 0) {
        if (!src) throw std::runtime_error("GdrContext::copy_from_host: src == nullptr");
        if (dst_offset + nbytes > size_)
            throw std::runtime_error("GdrContext::copy_from_host overflow");
        void* dst = static_cast<char*>(cpu_ptr_) + dst_offset;
        std::memcpy(dst, src, nbytes);
        this->h2d_sync();
    }
    
    inline void copy_from_host_tensor(const torch::Tensor& src, size_t nbytes) {
        if (nbytes == 0) {
            nbytes = src.nbytes();
        }
        this->copy_from_host(src.data_ptr(), nbytes);
    }

    inline void copy_from_host_int32(const std::vector<int>& src) {
        size_t nbytes = src.size() * sizeof(int);
        this->copy_from_host(src.data(), nbytes);
    }

    inline void copy_from_host_float(const std::vector<float>& src) {
        size_t nbytes = src.size() * sizeof(float);
        this->copy_from_host(src.data(), nbytes);
    }

    inline void copy_from_host_int64(const std::vector<int64_t>& src) {
        size_t nbytes = src.size() * sizeof(int64_t);
        this->copy_from_host(src.data(), nbytes);
    }

    /// Copy data back from GPU memory (BAR1 → host)
    inline void copy_to_host(void* dst, size_t nbytes, size_t src_offset = 0) {
        if (!dst) throw std::runtime_error("GdrContext::copy_to_host: dst == nullptr");
        if (src_offset + nbytes > size_)
            throw std::runtime_error("GdrContext::copy_to_host overflow");
        const void* src = static_cast<const char*>(cpu_ptr_) + src_offset;
        std::memcpy(dst, src, nbytes);
        this->d2h_sync();
    }

    inline void copy_to_host_tensor(torch::Tensor& dst, size_t nbytes) {
        if (nbytes == 0) {
            nbytes = dst.nbytes();
        }
        this->copy_to_host(dst.data_ptr(), nbytes);
    }

    inline std::vector<int> copy_to_host_int32(int nelems) {
        std::vector<int> result(nelems);
        this->copy_to_host(result.data(), nelems * sizeof(int));
        return result;
    }

    inline std::vector<float> copy_to_host_float(int nelems) {
        std::vector<float> result(nelems);
        this->copy_to_host(result.data(), nelems * sizeof(float));
        return result;
    }

    inline std::vector<int64_t> copy_to_host_int64(int nelems) {
        std::vector<int64_t> result(nelems);
        this->copy_to_host(result.data(), nelems * sizeof(int64_t));
        return result;
    }

    /// Fill GPU buffer with a byte value
    inline void fill(uint8_t value, size_t nbytes, size_t dst_offset = 0) {
        if (dst_offset + nbytes > size_)
            throw std::runtime_error("GdrContext::fill overflow");
        void* dst = static_cast<char*>(cpu_ptr_) + dst_offset;
        std::memset(dst, value, nbytes);
    }

    /// Get underlying GPU pointer (device virtual address)
    inline uintptr_t device_ptr() const { return dev_ptr_; }

    /// Get CPU-accessible BAR1 mapped pointer
    inline void* host_mapped_ptr() const { return cpu_ptr_; }

    /// Get total size of region (bytes)
    inline size_t size() const { return size_; }

    static void ensure_gdr_closed() {
        std::lock_guard<std::mutex> lock(gdr_mutex_);
        if (gdr_handle_) {
            gdr_close(gdr_handle_);
            gdr_handle_ = nullptr;
        }
    }

private:
    // ============================================================
    // Helpers
    // ============================================================
    void initialize(uint64_t dev_ptr_u64, size_t size) {
        ensure_gdr_open();

        CUdeviceptr dev_ptr = static_cast<CUdeviceptr>(dev_ptr_u64);

        // Pin the buffer - use exact pointer and size as provided
        int rc = gdr_pin_buffer(gdr_handle_, dev_ptr, size, 0, 0, &mh_);
        rc = gdr_map(gdr_handle_, mh_, &bar_ptr_, size);


        gdr_info_t info{};
        rc = gdr_get_info(gdr_handle_, mh_, &info);

        uint64_t aligned_va = info.va;
        size_t mapped_size = info.mapped_size > 0 ? info.mapped_size : size;
        
        uintptr_t va = static_cast<uintptr_t>(aligned_va);
        uintptr_t offset = dev_ptr_ - va;
        
        // Verify the original pointer is within the mapped range
        if (offset > mapped_size || dev_ptr_ < va || (dev_ptr_ - va) > mapped_size) {
            cleanup();
            std::ostringstream oss;
            oss << "GdrContext: dev_ptr not within mapped range"
                << " (dev_ptr=" << dev_ptr_
                << ", va=" << va
                << ", offset=" << offset
                << ", mapped_size=" << mapped_size << ")";
            throw std::runtime_error(oss.str());
        }
        
        // The CPU pointer is the mapped address plus the offset to reach the original pointer
        cpu_ptr_ = static_cast<void*>(static_cast<char*>(bar_ptr_) + offset);
    }

    void cleanup() noexcept {
        if (gdr_handle_) {
            if (bar_ptr_) {
                // Get the mapped size from info before unmapping
                gdr_unmap(gdr_handle_, mh_, bar_ptr_, size_);
                bar_ptr_ = nullptr;
            }
            if (mh_.h) {
                gdr_unpin_buffer(gdr_handle_, mh_);
                mh_.h = 0;
            }
        }
    }

    inline void h2d_sync() {
        _mm_sfence();
        CUDACHECK(cudaDeviceFlushGPUDirectRDMAWrites(
            cudaFlushGPUDirectRDMAWritesTarget::cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
            cudaFlushGPUDirectRDMAWritesScope::cudaFlushGPUDirectRDMAWritesToOwner
        ));
    }

    inline void d2h_sync() {
        _mm_lfence();
    }

    static void ensure_gdr_open() {
        std::lock_guard<std::mutex> lock(gdr_mutex_);
        if (!gdr_handle_) {
            gdr_handle_ = gdr_open();
            if (!gdr_handle_)
                throw std::runtime_error("gdr_open() failed (is gdrdrv loaded?)");
        }
    }

    // Static members for shared GDR handle
    inline static gdr_t gdr_handle_ = nullptr;
    inline static std::mutex gdr_mutex_;

    // Instance members
    gdr_mh_t mh_;
    void* bar_ptr_;
    void* cpu_ptr_;
    size_t size_;
    uintptr_t dev_ptr_;

    torch::Tensor tensor_;
};

using gdr_context_t = std::shared_ptr<GdrContext>;
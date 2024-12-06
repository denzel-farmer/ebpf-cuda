#include "MemoryPools.h"

PinnedMemoryPool::~PinnedMemoryPool() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    for (auto ptr : pinned_ptrs) {
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess) {
            std::cerr << "Failed to free pinned memory: " << cudaGetErrorString(err) << std::endl;
        }
    }
    pinned_ptrs.clear();
}

void* PinnedMemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    if (size == 0) {
        std::cerr << "Error: Attempting to allocate 0 bytes of pinned memory.\n";
        return nullptr;
    }
    
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << " (size: " << size << " bytes)\n";
        return nullptr;
    }
    
    pinned_ptrs.insert(ptr);
    return ptr;
}

void PinnedMemoryPool::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto it = pinned_ptrs.find(ptr);
    if (it != pinned_ptrs.end()) {
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess) {
            std::cerr << "cudaFreeHost failed: " << cudaGetErrorString(err) << std::endl;
        }
        pinned_ptrs.erase(it);
    }
    else {
        std::cerr << "Attempted to deallocate a pointer not in pinned pool: " << ptr << std::endl;
    }
}

NonPinnedMemoryPool::~NonPinnedMemoryPool() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    for (auto ptr : non_pinned_ptrs) {
        free(ptr);
    }
    non_pinned_ptrs.clear();
}

void* NonPinnedMemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    void* ptr = malloc(size);
    if (!ptr) {
        std::cerr << "Failed to allocate memory in NonPinnedMemoryPool" << std::endl;
        return nullptr;
    }
    non_pinned_ptrs.insert(ptr);
    return ptr;
}

void NonPinnedMemoryPool::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto it = non_pinned_ptrs.find(ptr);
    if (it != non_pinned_ptrs.end()) {
        free(ptr);
        non_pinned_ptrs.erase(it);
    }
    else {
        std::cerr << "Attempted to deallocate a pointer not in non-pinned pool: " << ptr << std::endl;
    }
}
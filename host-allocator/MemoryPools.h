#ifndef MEMORY_POOLS_H
#define MEMORY_POOLS_H

#include <cstddef>
#include <mutex>
#include <unordered_set>
#include <iostream>
#include <cuda_runtime.h>

class PinnedMemoryPool {
private:
    std::unordered_set<void*> pinned_ptrs;
    std::mutex pool_mutex;

public:
    PinnedMemoryPool() {}
    ~PinnedMemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr);
};

class NonPinnedMemoryPool {
private:
    std::unordered_set<void*> non_pinned_ptrs;
    std::mutex pool_mutex;

public:
    NonPinnedMemoryPool() {}
    ~NonPinnedMemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr);
};

#endif
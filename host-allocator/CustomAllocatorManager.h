// CustomAllocatorManager.h
#ifndef CUSTOM_ALLOCATOR_MANAGER_H
#define CUSTOM_ALLOCATOR_MANAGER_H

#include <mutex>
#include <unordered_map>
#include <string>
#include "MemoryPools.h"

class CustomAllocatorManager {
public:
    CustomAllocatorManager();

    void initialize(const std::string& mode);
    void* allocate_memory(size_t size);
    void deallocate_memory(void* ptr, size_t size);
    void load_frequency_data(const std::string& filename);
    void save_frequency_data(const std::string& filename);
    void update_frequency(void* return_addr);

private:
    PinnedMemoryPool pinned_pool;
    NonPinnedMemoryPool non_pinned_pool;
    std::unordered_map<void*, bool> allocation_type_map;
    std::unordered_map<void*, size_t> allocation_frequencies;
    std::mutex alloc_mutex;
    std::mutex freq_mutex;
};

// Global Allocator Manager Instance
extern CustomAllocatorManager g_allocator_manager;

// Global Allocation Functions
extern "C" void* allocate_memory(size_t size);
extern "C" void deallocate_memory(void* ptr, size_t size);

#endif
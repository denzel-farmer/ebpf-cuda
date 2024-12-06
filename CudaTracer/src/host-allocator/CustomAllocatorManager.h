// CustomAllocatorManager.h
#ifndef CUSTOM_ALLOCATOR_MANAGER_H
#define CUSTOM_ALLOCATOR_MANAGER_H

#include <mutex>
#include <unordered_map>
#include <string>
#include "MemoryPools.h"
#include "TracerAgent.h"

class CustomAllocatorManager {
public:
    CustomAllocatorManager();
    ~CustomAllocatorManager(){
        std::cout << "reached here" << std::endl;
        tracer_agent->DumpHistory("tracer_history.json", true);
    };
    void initialize(const std::string& mode);
    void* allocate_memory(size_t size);
    void deallocate_memory(void* ptr, size_t size);
    void load_tracer_history(const std::string& filename);
    void load_frequency_data(const std::string& filename);
    void save_frequency_data(const std::string& filename);
    void update_allocation_number(void* return_addr);
    void update_tracer_alloc(void* return_addr, size_t frequency, void* ptr, size_t size);
    void update_tracer_dealloc(void* ptr, size_t size);

private:
    PinnedMemoryPool pinned_pool;
    NonPinnedMemoryPool non_pinned_pool;
    std::unordered_map<void*, bool> allocation_type_map;
    std::unordered_map<void*, size_t> allocation_numbers;
    std::unordered_map<unsigned long, size_t> transfer_count_history;
    std::mutex alloc_mutex;
    std::mutex freq_mutex;
    unique_ptr<TracerAgent> tracer_agent;
    int tracer_history_used = 0;

};

// Global Allocator Manager Instance
extern CustomAllocatorManager g_allocator_manager;

// Global Allocation Functions
extern "C" void* allocate_memory(size_t size);
extern "C" void deallocate_memory(void* ptr, size_t size);

#endif
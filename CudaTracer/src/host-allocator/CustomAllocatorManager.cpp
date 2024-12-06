#include "CustomAllocatorManager.h"
#include <iostream>
#include <fstream>
#include <limits>

CustomAllocatorManager g_allocator_manager;

CustomAllocatorManager::CustomAllocatorManager()
    : pinned_pool(), non_pinned_pool() {}

void CustomAllocatorManager::initialize(const std::string& mode) {
    if (mode == "profile") {
        std::cout << "Initializing in Profiling Mode.\n";
    }
    else if (mode == "use") {
        std::cout << "Initializing in Optimized Mode.\n";
        load_frequency_data("frequency_data.txt");
    }
    else {
        std::cerr << "Unknown mode: " << mode << ". Use 'profile' or 'use'.\n";
    }
}

void* CustomAllocatorManager::allocate_memory(size_t size) {
    std::lock_guard<std::mutex> lock(alloc_mutex);

    void* return_addr = __builtin_return_address(0);

    std::cout << "Allocation called from return address: " << return_addr << std::endl;
    std::flush(std::cout);

    update_frequency(return_addr);

    bool use_pinned = false;
    {
        std::lock_guard<std::mutex> freq_lock(freq_mutex);
        auto it = allocation_frequencies.find(return_addr);
        if (it != allocation_frequencies.end()) {
            const size_t FREQUENCY_THRESHOLD = 5; // Adjust as needed
            if (it->second > FREQUENCY_THRESHOLD) {
                use_pinned = true;
            }
        }
    }

    void* ptr;
    if (use_pinned) {
        ptr = pinned_pool.allocate(size);
        {
            std::lock_guard<std::mutex> freq_lock(freq_mutex);
            allocation_type_map[ptr] = true;
        }
    }
    else {
        ptr = non_pinned_pool.allocate(size);
        {
            std::lock_guard<std::mutex> freq_lock(freq_mutex);
            allocation_type_map[ptr] = false;
        }
    }
    return ptr;
}

void CustomAllocatorManager::deallocate_memory(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(alloc_mutex);

    bool was_pinned = false;
    {
        std::lock_guard<std::mutex> freq_lock(freq_mutex);
        auto it = allocation_type_map.find(ptr);
        if (it != allocation_type_map.end()) {
            was_pinned = it->second;
            allocation_type_map.erase(it);
        }
        else {
            std::cerr << "Attempted to deallocate a pointer not tracked: " << ptr << std::endl;
            return;
        }
    }

    if (was_pinned) {
        pinned_pool.deallocate(ptr);
    }
    else {
        non_pinned_pool.deallocate(ptr);
    }
}

void CustomAllocatorManager::load_frequency_data(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open frequency data file: " << filename << ". Proceeding without frequency data.\n";
        return;
    }

    void* addr;
    size_t freq;
    while (infile >> addr >> freq) {
        if (infile.fail()) {
            std::cerr << "Error reading frequency data. Skipping line.\n";
            infile.clear();
            infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        allocation_frequencies[addr] = freq;
    }

    infile.close();
    std::cout << "Loaded frequency data from " << filename << ".\n";
}

void CustomAllocatorManager::save_frequency_data(const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open frequency data file for writing: " << filename << std::endl;
        return;
    }

    std::lock_guard<std::mutex> freq_lock(freq_mutex);
    for (const auto& pair : allocation_frequencies) {
        outfile << pair.first << " " << pair.second << "\n";
    }

    outfile.close();
    std::cout << "Saved frequency data to " << filename << ".\n";
}

void CustomAllocatorManager::update_frequency(void* return_addr) {
    std::lock_guard<std::mutex> freq_lock(freq_mutex);
    allocation_frequencies[return_addr]++;
}

extern "C" void* allocate_memory(size_t size) {
    return g_allocator_manager.allocate_memory(size);
}

extern "C" void deallocate_memory(void* ptr, size_t size) {
    g_allocator_manager.deallocate_memory(ptr, size);
}
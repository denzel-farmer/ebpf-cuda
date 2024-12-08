#include "CustomAllocatorManager.h"
#include <iostream>
#include <fstream>
#include <limits>
#include "AllocationHistory.h"
#include <chrono>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

CustomAllocatorManager g_allocator_manager;

CustomAllocatorManager::CustomAllocatorManager()
    : pinned_pool(), non_pinned_pool() {
        tracer_agent = make_unique<TracerAgent>();
    }

void CustomAllocatorManager::initialize(const std::string& mode) {
    // Reset number of allocations for each call site, allows reinitialization
    reset_allocation_numbers();
    tracer_agent->StopAgent(); 
    if (mode == "profile") {
         tracer_agent->StartAgentAsync();
        std::cout << "Initializing in Profiling Mode.\n";
    }
    else if (mode == "use") {
        std::cout << "Initializing in Optimized Mode.\n";
        load_tracer_history("tracer_history.json");
    }
    else {
        std::cerr << "Unknown mode: " << mode << ". Use 'profile' or 'use'.\n";
    }
}

void* CustomAllocatorManager::allocate_memory(size_t size) {
    std::lock_guard<std::mutex> lock(alloc_mutex);

    void* return_addr = __builtin_return_address(0);

  //  std::cout << "Allocation called from return address: " << return_addr << std::endl;
    std::flush(std::cout);

    update_allocation_number(return_addr);

    bool use_pinned = false;
    // default to use pinning if no tracer history used
    if (tracer_history_used == 0){
        use_pinned = true;
    } else {
        std::lock_guard<std::mutex> freq_lock(freq_mutex);
        auto it = transfer_count_history.find(((unsigned long)return_addr << 16) + allocation_numbers[return_addr]);
        if (it != transfer_count_history.end()) {
            const size_t FREQUENCY_THRESHOLD = 5; // Adjust as needed
            if (it->second > FREQUENCY_THRESHOLD) {
                use_pinned = true;
            }
        }
    }

    void* ptr;
    if (use_pinned) {
        ptr = pinned_pool.allocate(size);
        total_amount_pinned += size;
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

    update_tracer_alloc(return_addr, allocation_numbers[return_addr], ptr, size);

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

    update_tracer_dealloc(ptr, size);
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
        allocation_numbers[addr] = freq;
    }

    infile.close();
    std::cout << "Loaded frequency data from " << filename << ".\n";
}

void CustomAllocatorManager::load_tracer_history(const std::string& filename){
    // std::ifstream infile(filename);
    // if (!infile.is_open()) {
    //     std::cerr << "Failed to open tracer data file: " << filename << ". Proceeding without tracer data.\n";
    //     return;
    // } else{
    //     std::cout << "tracer history load worked" << std::endl;
    // }

    boost::property_tree::ptree pt;
    try {
        boost::property_tree::read_json(filename, pt);
    } catch (const boost::property_tree::json_parser_error& e) {
        std::cerr << "Error reading JSON file: " << e.what() << std::endl;
        return;
    }

    for (const auto& allocation : pt.get_child("Allocations")) {
        try {
            std::string call_site_str = allocation.second.get<std::string>("AllocTag.call_site");
            unsigned long call_site = std::stoul(call_site_str);

            std::string call_no_str = allocation.second.get<std::string>("AllocTag.call_no");
            unsigned long call_no = std::stoul(call_no_str);

            std::string transfer_count_str = allocation.second.get<std::string>("transfer_count");
            unsigned long transfer_count = std::stoul(transfer_count_str);

            unsigned long unique_identifier = (call_site << 16) + call_no;

            transfer_count_history[unique_identifier] = transfer_count;
        }catch(const boost::property_tree::ptree_bad_data& e) {
            std::cerr << "Error accessing JSON key: " << e.what() << std::endl;
        }
    }
    
    // infile.close();
    std::cout << "Loaded tracer data from " << filename << ".\n";
    tracer_history_used = 1;

}

void CustomAllocatorManager::save_frequency_data(const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open frequency data file for writing: " << filename << std::endl;
        return;
    }

    std::lock_guard<std::mutex> freq_lock(freq_mutex);
    for (const auto& pair : allocation_numbers) {
        outfile << pair.first << " " << pair.second << "\n";
    }

    outfile.close();
    std::cout << "Saved frequency data to " << filename << ".\n";
}

void CustomAllocatorManager::reset_allocation_numbers() {
    std::lock_guard<std::mutex> freq_lock(freq_mutex);
    allocation_numbers.clear();
}

void CustomAllocatorManager::update_allocation_number(void* return_addr) {
    std::lock_guard<std::mutex> freq_lock(freq_mutex);
    allocation_numbers[return_addr]++;
}

unsigned long get_time_since_boot_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_BOOTTIME, &ts) != 0) {
        perror("clock_gettime");
        return 0; // Return 0 or handle the error as needed
    }
    return static_cast<unsigned long>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

void CustomAllocatorManager::update_tracer_alloc(void* return_addr, size_t frequency, void* ptr, size_t size){
    AllocationIdentifier allocation_identifier((unsigned long) return_addr, (unsigned long) frequency);
    AllocationRange allocation_range( (unsigned long) ptr, static_cast<unsigned long>(size));
    auto timestamp = get_time_since_boot_ns();
    EventInfo event_info(timestamp, EventType::ALLOC);
    AllocationEvent event(allocation_range, event_info);
    tracer_agent->HandleEvent(event, allocation_identifier);
}

void CustomAllocatorManager::update_tracer_dealloc(void* ptr, size_t size){
    AllocationRange allocation_range((unsigned long) ptr, static_cast<unsigned long>(size));
    auto timestamp = get_time_since_boot_ns();
    EventInfo event_info(timestamp, EventType::FREE);
    AllocationEvent event(allocation_range, event_info);
    tracer_agent->HandleEvent(event);
}

extern "C" void* allocate_memory(size_t size) {
    return g_allocator_manager.allocate_memory(size);
}

extern "C" void deallocate_memory(void* ptr, size_t size) {
    g_allocator_manager.deallocate_memory(ptr, size);
}
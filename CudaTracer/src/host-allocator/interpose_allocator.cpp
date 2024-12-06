
#include "CustomAllocatorManager.h"
#include <cstring> // For memset
#include <string>
#include <chrono>

using namespace std::chrono;

void perform_allocations(int iterations, size_t size) {
    for (int i = 0; i < iterations; ++i) {
        void* ptr = CustomAllocatorManager::getInstance().allocate_memory(size);
        if (ptr) {
            for (int j = 0; j < 1000; ++j){
                memset(ptr, 'A', size);
            }
            CustomAllocatorManager::getInstance().deallocate_memory(ptr, size);
        }
        else {
            std::cerr << "Memory allocation failed at iteration " << i << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mode>\n";
        std::cerr << "Modes:\n";
        std::cerr << "  profile - Track allocation frequencies and save to file.\n";
        std::cerr << "  use     - Load frequency data and optimize allocations.\n";
        return 1;
    }

    std::string mode = argv[1];
    if (mode != "profile" && mode != "use") {
        std::cerr << "Invalid mode: " << mode << ". Use 'profile' or 'use'.\n";
        return 1;
    }

    CustomAllocatorManager::getInstance().initialize(mode);

    auto t1 = std::chrono::system_clock::now();

    if (mode == "profile") {
        perform_allocations(10, 1024 * 1024 * 16);
        // perform_allocations(10, 2048 );

        CustomAllocatorManager::getInstance().save_frequency_data("frequency_data.txt");
    }
    else if (mode == "use") {
        perform_allocations(10, 1024 * 1024 * 16);
        // perform_allocations(10, 2048);
    }

    auto t2 = std::chrono::system_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds.\n";

    return 0;
}
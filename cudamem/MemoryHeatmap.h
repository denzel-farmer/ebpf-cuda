#pragma once

#include <map>
#include <string>

constexpr size_t PAGE_SIZE = 4096;
constexpr unsigned long MAX_TRACKED = (1 << 23);
constexpr unsigned long PAGE_MASK = ~(PAGE_SIZE - 1);
constexpr size_t MAX_PRINT_PAGES = (4096*4);

// Helper to convert address to page number
inline unsigned long GetPageNumber(unsigned long address) {
    return address / PAGE_SIZE;
}

// Helper to convert page number to address
inline unsigned long GetPageAddress(unsigned long page_number) {
    return page_number * PAGE_SIZE;
}


class MemoryHeatmap {
public:
    MemoryHeatmap() {
        MemoryMap = std::map<unsigned long, size_t>();
    }

    // Record a new access event 
    void RecordAccess(unsigned long address, size_t size);

    size_t GetPageAccessCount(unsigned long page_number);
    unsigned long GetMinPageNum();
    unsigned long GetMaxPageNum();

    // Print heatmap of memory accesses
    void PrintHeatmap(size_t cols);

private:
    // Current representation of memory: map from page number to access count
    // Very inefficient, should just use sparse ranges
    std::map<unsigned long, size_t> MemoryMap;
};
#include <iostream>
#include <map>

#include <iomanip>
#include "MemoryHeatmap.h"

using namespace std;

void MemoryHeatmap::RecordAccess(unsigned long address, size_t size) {
    if (MemoryMap.size() > MAX_TRACKED) {
        return;
    }
    if (size == 0) {
        return;
    }

    // Iterate over all pages in the range
    unsigned long page_number = GetPageNumber(address);
    unsigned long end_page_number = GetPageNumber(address + size);
    for (unsigned long i = page_number; i <= end_page_number; i++) {
        if (MemoryMap.find(i) == MemoryMap.end()) {
            MemoryMap[i] = 1;
        } else {
            MemoryMap[i]++;
        }
    }
}

size_t MemoryHeatmap::GetPageAccessCount(unsigned long page_number) {
    if (MemoryMap.find(page_number) == MemoryMap.end()) {
        return 0;
    }
    return MemoryMap[page_number];
}

unsigned long MemoryHeatmap::GetMinPageNum() {
    if (MemoryMap.empty()) {
        return 0;
    }
    return MemoryMap.begin()->first;
}

unsigned long MemoryHeatmap::GetMaxPageNum() {
    if (MemoryMap.empty()) {
        return 0;
    }
    return MemoryMap.rbegin()->first;
}



// Print heatmap of memory accesses
void MemoryHeatmap::PrintHeatmap(size_t cols) {
    cout << "Printing heatmap" << endl;

    // // Print 1d heatmap
    // for (auto const& [key, val] : MemoryMap) {
    //     cout << key << ": " << val << endl;
    // }

    // Print 2d heatmap
    unsigned long min_page = GetMinPageNum();
    unsigned long max_page = GetMaxPageNum();

    bool print_sparse = false;
    if (max_page - min_page > MAX_PRINT_PAGES) {
        cout << "Memory heatmap too large to print, printing sparse heatmap" << endl;
        print_sparse = true;
    }

    cout << "min: " << hex << GetPageAddress(min_page) << endl;
    cout << "max: " << hex << GetPageAddress(max_page)-1 << endl;

    size_t count = 0;
    for (unsigned long i = min_page; i <= max_page; i++) {
        if (count >= MAX_PRINT_PAGES) {
            cout << "Reached max print pages" << endl;
            break;
        }
    
        if (count % cols == 0) {
            cout << endl;
            cout << hex << GetPageAddress(i) << ": ";
        }

        size_t access_count = GetPageAccessCount(i);
        if (print_sparse && access_count == 0) {
            continue;
        }

        cout << dec << setw(3) << setfill(' ') << access_count << " ";
        count++;
    }

    cout << endl;
    // unsigned long num_pages = max_page - min_page + 1;
    // unsigned long num_rows = num_pages / 64;
    // unsigned long num_cols = 64;
    // unsigned long page_number = min_page;

    // string heatmap = "";
    // for (unsigned long i = 0; i < num_rows; i++) {
    //     for (unsigned long j = 0; j < num_cols; j++) {
    //         string padded_page = "";
    //         unsigned long page = page_number + i * num_cols + j;
    //         if (MemoryMap.find(page) == MemoryMap.end()) {
    //             padded_page += "0";
    //         } else {
    //             padded_page += to_string(MemoryMap[page]);
    //         }

    //         // Pad to 5 characters
    //         while (padded_page.size() < 5) {
    //             padded_page = " " + padded_page;
    //         }
    //         heatmap += padded_page;
    //     }
    //     heatmap += "\n";
    // }

    // return heatmap;
}

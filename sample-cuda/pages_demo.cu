#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h> // For getpid()

#define PAGE_SIZE 4096
#define NUM_PAGES 16
#define ARRAY_SIZE (PAGE_SIZE * NUM_PAGES)

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void waitForUser() {
    pid_t pid = getpid();
    std::cout << "Process ID (PID): " << pid << std::endl;
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.get();
}

void copyAllPages(char* d_dst, char* d_src) {
    std::cout << "Copying all pages at once." << std::endl;
    checkCudaError(cudaMemcpy(d_dst, d_src, ARRAY_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy all pages");
}

void copyFirstEightPages(char* d_dst, char* d_src) {
    std::cout << "Copying first 8 pages." << std::endl;
    size_t size = PAGE_SIZE * 8;
    checkCudaError(cudaMemcpy(d_dst, d_src, size, cudaMemcpyHostToDevice), "cudaMemcpy first 8 pages");
}

void copyEveryOtherPage(char* d_dst, char* d_src) {
    std::cout << "Copying every other page." << std::endl;
    for (int i = 0; i < NUM_PAGES; i += 2) {
        char* dst_ptr = d_dst + i * PAGE_SIZE;
        char* src_ptr = d_src + i * PAGE_SIZE;
        checkCudaError(cudaMemcpy(dst_ptr, src_ptr, PAGE_SIZE / 2, cudaMemcpyHostToDevice), "cudaMemcpy every other page");
    }
}

void copyLastEightPages(char* d_dst, char* d_src) {
    std::cout << "Copying last 8 pages." << std::endl;
    size_t offset = PAGE_SIZE * 8;
    size_t size = PAGE_SIZE * 8;
    checkCudaError(cudaMemcpy(d_dst + offset, d_src + offset, size, cudaMemcpyHostToDevice), "cudaMemcpy last 8 pages");
}

void copyPagesInReverse(char* d_dst, char* d_src) {
    std::cout << "Copying pages in reverse order." << std::endl;
    for (int i = NUM_PAGES - 1; i >= 0; --i) {
        char* dst_ptr = d_dst + i * PAGE_SIZE;
        char* src_ptr = d_src + i * PAGE_SIZE;
        checkCudaError(cudaMemcpy(dst_ptr, src_ptr, PAGE_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy reverse order");
    }
}

int main() {
    // Allocate memory on the device
    char* d_src = nullptr;
    char* d_dst = nullptr;
    checkCudaError(cudaMalloc(&d_src, ARRAY_SIZE), "cudaMalloc d_src");
    checkCudaError(cudaMalloc(&d_dst, ARRAY_SIZE), "cudaMalloc d_dst");

    // Initialize source data (optional)
    checkCudaError(cudaMemset(d_src, 1, ARRAY_SIZE), "cudaMemset d_src");
    // Copy all pages at once
    std::cout << "Test: Copy all pages at once" << std::endl;
    waitForUser();
    copyAllPages(d_dst, d_src);

    // Copy only the first 8 pages
    std::cout << "Test: Copy only the first 8 pages" << std::endl;
    waitForUser();
    copyFirstEightPages(d_dst, d_src);

    // Copy every other page
    std::cout << "Test: Copy every other page" << std::endl;
    waitForUser();
    copyEveryOtherPage(d_dst, d_src);

    // Copy the last 8 pages
    std::cout << "Test: Copy the last 8 pages" << std::endl;
    waitForUser();
    copyLastEightPages(d_dst, d_src);

    // Copy pages in reverse order
    std::cout << "Test: Copy pages in reverse order" << std::endl;
    waitForUser();
    copyPagesInReverse(d_dst, d_src);

    // Clean up
    checkCudaError(cudaFree(d_src), "cudaFree d_src");
    checkCudaError(cudaFree(d_dst), "cudaFree d_dst");

    std::cout << "Program completed successfully." << std::endl;
    return 0;
}

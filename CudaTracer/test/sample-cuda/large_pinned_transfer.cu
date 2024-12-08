#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h> // For getpid()
#include <sys/mman.h>

#define PAGE_SIZE 4096

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


void largeTransfer(size_t size) {
    // Use CudaHostAlloc to allocate pinned memory
    char* h_src = nullptr;
    char* h_dst = nullptr;

    std::cout << "[largeTransfer] Allocating " << size << " bytes of pinned memory." << std::endl;
    waitForUser();

   checkCudaError(cudaHostAlloc(&h_src, size, cudaHostAllocDefault), "cudaHostAlloc h_src");
    // h_src = (char*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    // if (h_src == MAP_FAILED) {
    //     std::cerr << "Error: Unable to allocate host memory using mmap." << std::endl;
    //     exit(EXIT_FAILURE);
    // }
    // Wait for user input
    std::cout << "[largeTransfer] memory allocated but not used" << std::endl;
    waitForUser();

    // Touch each page to ensure it is allocated
    for (size_t i = 0; i < size; i += PAGE_SIZE) {
        h_src[i] = 0x52;
    }

    // Wait for user input
    std::cout << "[largeTransfer] memory allocated and used" << std::endl;
    waitForUser();

    // Transfer data from host to device
    char* d_src = nullptr;
    checkCudaError(cudaMalloc(&d_src, size), "cudaMalloc d_src");
    checkCudaError(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice), "cudaMemcpy h_src to d_src");

    // Wait for user input
    std::cout << "[largeTransfer] data transferred to device, ready to free" << std::endl;
    waitForUser();

    // Free memory
    checkCudaError(cudaFree(d_src), "cudaFree d_src");
    checkCudaError(cudaFreeHost(h_src), "cudaFreeHost h_src");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size_in_MB>" << std::endl;
        return EXIT_FAILURE;
    }

    size_t sizeInMB = std::stoul(argv[1]);
    size_t sizeInBytes = sizeInMB * 1024 * 1024;

    largeTransfer(sizeInBytes);

    std::cout << "Program completed successfully." << std::endl;
    return 0;
}

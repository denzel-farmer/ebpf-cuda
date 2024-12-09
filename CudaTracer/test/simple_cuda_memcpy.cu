#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h>

__global__ void simpleKernel(int *d_data) {
    // Simple kernel that does nothing
}

void dummyFunction() {
    void *return_address = __builtin_return_address(0);
    std::cout << "Return address: " << return_address << std::endl;
}

int main() {
    void *h_data;
    int *d_data;


    // Print the process ID
    std::cout << "Process ID: " << getpid() << std::endl;
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.get();

    // Allocate device memory
    cudaMalloc((void**)&d_data, sizeof(int));

    // Allocate pinned host memory
    cudaHostAlloc(&h_data, sizeof(int), cudaHostAllocDefault);

    // Copy data from host to device
    dummyFunction();
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
    // Print the instruction pointer
    std::cout << "Instruction pointer: " << __builtin_return_address(0) << std::endl;

    // Launch a simple kernel
    simpleKernel<<<1, 1>>>(d_data);

    // synchronize
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_data);

    // Free pinned host memory
    cudaFreeHost(h_data);

    return 0;
}
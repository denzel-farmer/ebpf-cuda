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
    int h_data = 42;
    int *d_data;


    // Print the process ID
    std::cout << "Process ID: " << getpid() << std::endl;
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.get();

    // Allocate device memory
    cudaMalloc((void**)&d_data, sizeof(int));

    // Copy data from host to device
    dummyFunction();
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
    // Print the instruction pointer
    std::cout << "Instruction pointer: " << __builtin_return_address(0) << std::endl;

    // Launch a simple kernel
    simpleKernel<<<1, 1>>>(d_data);

    // Free device memory
    cudaFree(d_data);

    return 0;
}
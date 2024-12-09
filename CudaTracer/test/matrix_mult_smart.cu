#include "CustomAllocatorManager.h"

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

constexpr int NUM = 1024;
constexpr int ITERS = 512;
constexpr int matrix_size = NUM * NUM * sizeof(double);

__global__ void matMul(const double *A, const double *B, double *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C to compute
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C to compute

    if(row < NUM && col < NUM){
        double value = 0.0f;
        for(int k = 0; k < NUM; k++){
            value += A[row * NUM + k] * B[k * NUM + col];
        }
        C[row * NUM + col] = value;
    }
}


void saveFinalResult(double* matrix) {
    // Calculate the checksum of the matrix
    double checksum = 0.0f;
    for (int i = 0; i < NUM * NUM; i++) {
        checksum += matrix[i];
    }
    std::cout << "Checksum of the final result: " << checksum << std::endl;
}

void updateIntermediateResult(double* matrix) {
    matrix[0] += 0.1f;
}


void perform_test() {
    double *d_A, *d_B, *d_C, *d_D, *d_E;
    double *h_A, *h_B, *h_C, *h_D, *h_E;

    // Device-side memory allocation (ommits error checking)
    cudaMalloc((void**)&d_A, matrix_size);
    cudaMalloc((void**)&d_B, matrix_size);
    cudaMalloc((void**)&d_C, matrix_size);
    cudaMalloc((void**)&d_D, matrix_size);
    cudaMalloc((void**)&d_E, matrix_size);

    // Host-side memory allocation using custom allocator
    h_A = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));
    h_B = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));
    h_C = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));
    h_D = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));
    h_E = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));

    
    // Initialize matrices with 1s for h_A, h_B, and h_D
    for(int i = 0; i < NUM; i++){
        for(int j = 0; j < NUM; j++){
            h_A[i * NUM + j] = 1.0;
            h_B[i * NUM + j] = 1.0;
            h_D[i * NUM + j] = 1.0;
        }
    }

    // Synchronize to ensure timing is accurate
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    // Transfer from unpinned memory to the device (slow)
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(32, 32); // 32x32 threads per block
    dim3 gridDim((NUM + blockDim.x - 1) / blockDim.x, (NUM + blockDim.y - 1) / blockDim.y); // Grid size to cover the entire matrix

    // Perform first operation, C = A x B
    matMul<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    // Transfer result back (assuming intermediate results modified by processIntermediateResult)
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
    updateIntermediateResult(h_C);

    // Transfer matrix D to device 
    cudaMemcpy(d_D, h_D, matrix_size, cudaMemcpyHostToDevice);
    for (int i = 0; i < ITERS; i++) {
        
        // Transfer matrix C to devce
        cudaMemcpy(d_C, h_C, matrix_size, cudaMemcpyHostToDevice);

        // Perform repeat operation, E = C x D
        matMul<<<gridDim, blockDim>>>(d_C, d_D, d_E);

        // Move result from d_E back to h_C, save result, and repeat
        if (i != (ITERS - 1)) {
            cudaMemcpy(h_C, d_E, matrix_size, cudaMemcpyDeviceToHost);
            updateIntermediateResult(h_C);
        }
    }

    // Move final result 
    cudaMemcpy(h_E, d_E, matrix_size, cudaMemcpyDeviceToHost);

    // Synchronize and end timing
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and print the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Duration: " << duration.count() << " ms" << std::endl;

    saveFinalResult(h_E);

    // Clean up device memory 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);

    // Clean up host memory
    g_allocator_manager.deallocate_memory(h_A, matrix_size);
    g_allocator_manager.deallocate_memory(h_B, matrix_size);
    g_allocator_manager.deallocate_memory(h_C, matrix_size);
    g_allocator_manager.deallocate_memory(h_D, matrix_size);
    g_allocator_manager.deallocate_memory(h_E, matrix_size);
}


int main() {

    // First run in profile mode to generate the history file
    g_allocator_manager.initialize("profile"); 
    perform_test();

    // Run in use mode
    g_allocator_manager.initialize("use", false);
    perform_test();

}
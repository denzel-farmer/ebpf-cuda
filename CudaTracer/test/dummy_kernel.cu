// dummy_kernel.cu
#include <cuda_runtime.h>
#include <cstddef>

__global__ void dummy_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1.0f;
    }
}
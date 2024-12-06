#ifndef DUMMY_KERNEL_H
#define DUMMY_KERNEL_H

#include <cuda_runtime.h>

__global__ void dummy_kernel(float* data, size_t size);

#endif
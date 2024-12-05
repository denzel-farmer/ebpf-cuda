#pragma once
#include <sys/types.h>

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

struct CudaProcessInfo {
    pid_t pid;
};

struct CudaTransferEvent {
    void *destination;
    const void *source;
    size_t size;
    enum cudaMemcpyKind direction;
    struct CudaProcessInfo processInfo;
};

struct CudaMemcpyEvent {
    void *source;
    const void *destination;
    size_t size;
    cudaMemcpyKind direction;
    CudaProcessInfo processInfo;
};
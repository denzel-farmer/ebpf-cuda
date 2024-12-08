
#ifndef CUDA_EVENTS_H
#define CUDA_EVENTS_H


#ifndef __CUDA_RUNTIME_H__
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};
#endif // __CUDA_RUNTIME_H__


struct CudaProcessInfo {
    pid_t pid;
};

// struct CudaTransferEvent {
//     void *destination;
//     const void *source;
//     size_t size;
//     enum cudaMemcpyKind direction;
//     struct CudaProcessInfo processInfo;
// };

struct CudaMemcpyEvent {
    unsigned long source;
    unsigned long destination;
    unsigned long return_address;
    unsigned long long timestamp;
    size_t size;
    enum cudaMemcpyKind direction;
    struct CudaProcessInfo processInfo;
};

// struct MallocEvent {
//     unsigned long address;
//     size_t size;
//     unsigned long return_address;
//     unsigned long long timestamp;
//     struct CudaProcessInfo processInfo;
// }

#endif // CUDA_EVENTS_H
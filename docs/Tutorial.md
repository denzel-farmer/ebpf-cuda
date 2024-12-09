# 1. The TracerAgent Library

## 1.1 The TestTracerAgent Executable
For an example usage of the TracerAgent library, we can look at the implementation 
of the `CudaTracer/test/TestTracerAgent.cu` executable.

The executable's main function is very simple. It first takes in the `pid` argument,
then initializes a tracer with that argument and blocks.

```C++
int main() {
    int pid = std::stoi(argv[1]);

    // Register signal handler
    signal(SIGINT, signalHandler);

    TracerAgent agent(pid);

    std::cout << "TracerAgent created, press enter to start" << std::endl;

    std::cin.get();

    /* ... */
}

```

Then, once the user presses enter, the tracer calls `StartAgentAsync` which begins profiling,
and blocks again.
```C++
int main() {

    /* ... */

    agent.StartAgentAsync();

    std::cout << "Agent started. Press Enter to dump history" << std::endl;
    std::cin.get();

    /* ... */

}

```

Now, once the target PID is finished, the user can press enter and dump the database to 
a human-readable JSON format, and wait for a final input to stop the agent and shutdown.

```C++
int main() {
    
    /* ... */

    agent.DumpHistory("history_dump.json", DumpFormat::JSON, true);
    std::cout << "History dumped to history_dump.json" << std::endl;

    std::cout << "Press Enter to stop agent" << std::endl;
    std::cin.get();
    agent.StopAgent();

    return 0;

}

```
This example is already included in the makefile, so it will be compiled with `make all`.
```bash
make clean
make all
```

This also builds `simple_cuda_memcpy` from `CudaTracer/test/simple_cuda_memcpy.cu`, which 
reports its PID, waits for input, then does a sequence of allocation, transfer to device, 
transfer from device, and free.

To demonstrate, first launch `simple_cuda_memcpy` in a separate shell.

```bash
sudo ./simple_cuda_memcpy
```

It will report its PID, which can then be used as an argument to `TestTracerAgent`.

```bash
sudo ./TestTracerAgent <pid>
```

Then, press enter in `TestTracerAgent` to start tracing, return to `simple_cuda_memcpy`,
and allow it to run to completion.

`TestTracerAgent` records all the events, which it dumps (in verbose mode, so including
all events) to `history_dump.json`. 

## 1.2 The TracerAgent Library

To see an example of the TracerAgent used as a library, see the [custom allocator design](Design.md).

# 2. The Smart-Pinning Allocator 

In this tutorial, we will walk through the simple task of applying our smart-pinning allocator
to a matrix multiplication task. 

## 2.1 The Kernel
First, we need a kernel to use with the host-side memory we allocate. In this tutorial, we'll use 
simple matrix multiplication:
```C++
__global__ void matMul(const float *A, const float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C to compute
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C to compute

    if(row < N && col < N){
        float value = 0.0f;
        for(int k = 0; k < N; k++){
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
```
## 2.2 Unpinned Example

Now, we can use this kernel with unpinned memory, allocated with `malloc`, to 
multiply a sample matrix A times matrix B, and then multiply the result (C) by a
final matrix D repeatedly.

First, include some headers and define constants.
```C++

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

constexpr int NUM = 1024;
constexpr int ITERS = 512;
constexpr int size = NUM * NUM * sizeof(double);
```

Then, write a helper function that calculates a checksum of a NUM by NUM array,
so we can see the final result is affected by our operation.

```C++
void saveFinalResult(double* matrix) {
    // Calculate the checksum of the matrix
    double checksum = 0.0f;
    for (int i = 0; i < NUM * NUM; i++) {
        checksum += matrix[i];
    }
    std::cout << "Checksum of the final result: " << checksum << std::endl;
}
```

We also might want to modify the matrix between operation. This will require us
to copy it to the host device aftere each stage, rather than doing all computation
on the device.

For this example, we can just add 0.1 to the first element of the matrix.

```C++
void updateIntermediateResult(double* matrix) {
    matrix[0] += 0.1f;
}

```

Now, we can write the main function that actually does computation. 

First, declare pointers and allocate device-side memory (not affected by pinning).

```C++
int main() {
    double *d_A, *d_B, *d_C, *d_D, *d_E;
    double *h_A, *h_B, *h_C, *h_D, *h_E;

    // Device-side memory allocation (ommits error checking)
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_D, size);
    cudaMalloc((void**)&d_E, size);

    /* ... */
}
```

Then, allocate the host-side memory for each matrix (for now, unpinned) and initialize the input 
matricies A, B, and D with dummy values.

```C++
int main () {
    
    /* ... */

    // Host-side memory allocation
    h_A = (double *)malloc(size);
    h_B = (double *)malloc(size);
    h_C = (double *)malloc(size);

    h_D = (double *)malloc(size);
    h_E = (double *)malloc(size);
    
    // Initialize matrices with 1s for h_A, h_B, and h_D
    for(int i = 0; i < NUM; i++){
        for(int j = 0; j < NUM; j++){
            h_A[i * NUM + j] = 1.0;
            h_B[i * NUM + j] = 1.0;
            h_D[i * NUM + j] = 1.0;
        }
    }

    /* ... */

}

````

Now, we can synchronize, begin a timer, and perform the first operation.
```C++
int main() {

    /* ... */

   // Synchronize to ensure timing is accurate
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    // Transfer from unpinned memory to the device (slow)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(32, 32); // 32x32 threads per block
    dim3 gridDim((NUM + blockDim.x - 1) / blockDim.x, (NUM + blockDim.y - 1) / blockDim.y); // Grid size to cover the entire matrix

    // Perform first operation, C = A x B
    matMul<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    // Transfer result back (assuming intermediate results modified by processIntermediateResult)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    updateIntermediateResult(h_C);

    /* ... */

}

```
The first operation is done, and we have paid a small price for using unpinned memory 
for the tensors each time we transfer. However, the next operations performs 64 transfers of matrix C,
which *repeatedly* incurs this overhead (a 'hotspot').

```C++

int main {
    /* ... */

    // Transfer matrix D to device 
    cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice);
    for (int i = 0; i < ITERS; i++) {
        
        // Transfer matrix C to devce
        cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

        // Perform repeat operation, E = C x D
        matMul<<<gridDim, blockDim>>>(d_C, d_D, d_E);

        // Move result from d_E back to h_C, save result, and repeat
        if (i != (ITERS - 1)) {
            cudaMemcpy(h_C, d_E, size, cudaMemcpyDeviceToHost);
            updateIntermediateResult(h_C);
        }
    }

    // Move final result 
    cudaMemcpy(h_E, d_E, size, cudaMemcpyDeviceToHost);

    // Synchronize and end timing
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and print the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Duration: " << duration.count() << " ms" << std::endl;

    saveFinalResult(h_E);

    /* ... */

}

```

And finally, free up resources.

```C++
int main() {
    
    /* ... */

    // Clean up device memory 
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);
    cudaFree(h_D);
    cudaFree(h_E);

    // Clean up host memory
    free(h_A);
    free(h_B);
    free(h_C);   
    free(h_D);
    free(h_E);
}
```

While not optimal, this example demonstrates a simple pattern, where the majority 
of allocations see only 1 or 2 transfers, but a single allocation (host matrix C)
represents a hotspot. 

To compile, we can use save the example to `matrix_mult_unpinned.cu` and use `nvcc`:
```bash
nvcc -O2 test/matrix_mult_unpinned.cu -o matrix_mult_unpinned
```

Then, running should print out timing details for the operation. Note that the
final result is likely `inf`, because it is too large for the `double` without more 
carefully selecting the inital values and update function.

```bash 
./matrx_mult_unpinned
```

On my test machine, I get the following:
```
Duration: 4155 ms
Checksum of the final result: inf
```

It is likely worth pinning only matrix C. First, though, we measure the 
maximum possible performance by pinning all allocations.

## 2.3 Fully Pinned Example

To switch to full pinning, only the memory allocation and deallocation lines must 
be replaced with `cudaHostAlloc` and `cudaHostFree`:

```C++
int main() {

    /* ... */

    // Host-side memory allocation using cudaHostAlloc
    cudaHostAlloc((void**)&h_A, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_C, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_D, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_E, size, cudaHostAllocDefault);

    /* ... */

    // Clean up host memory
    cudaHostFree(h_A);
    cudaHostFree(h_B);
    cudaHostFree(h_C);
    cudaHostFree(h_D);
    cudaHostFree(h_E);

    /* ... */

}
```

As before, we can save to a file (`matrix_mult_pinned.cu`) and compile and run:
```bash
nvcc -O2 test/matrix_mult_pinned.cu -o matrix_mult_pinned
./matrix_mult_pinned
```
As expected, the duration is significantly smaller:
```
Duration: 3137 ms
Checksum of the final result: inf
```

While the duration for this sample will be smaller than in example 2.2, it 
probably need only pins allocation C to see most of the performance benefit.

In this case, the programmer likely knows this and could do so manually. However,
our custom allocator can make the same decision 

## 2.4 Smart Pinning Example

To use the smart allocator, we just need to include the required header,. `CustomAllocatorManager.h`.

```C++
#include "CustomAllocatorManager.h"
```

Now, to demonstrate the benefit of the allocator, we first must do a `profile` run, then 
a `use` run. So, we will move our test from function `main` to `perform_test`. 

Then, again replace host-side allocation and deallocation.

```C++
int perform_test() {

    /* ... */

    // Host-side memory allocation using custom allocator
    h_A = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));
    h_B = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));
    h_C = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));
    h_D = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));
    h_E = static_cast<double *>(g_allocator_manager.allocate_memory(matrix_size));

    

    /* ... */

    // Clean up host memory
    g_allocator_manager.deallocate_memory(h_A, matrix_size);
    g_allocator_manager.deallocate_memory(h_B, matrix_size);
    g_allocator_manager.deallocate_memory(h_C, matrix_size);
    g_allocator_manager.deallocate_memory(h_D, matrix_size);
    g_allocator_manager.deallocate_memory(h_E, matrix_size);

    /* ... */

}
```

From main, we call perform_test twice, first to profile and then to use.

```C++
int main() {

    // First run in profile mode to generate the history file
    g_allocator_manager.initialize("profile"); 
    perform_test();

    // Now run in use mode
    g_allocator_manager.initialize("use");
    perform_test();

}
```

Since this example requires the custom allocator sources, we should compile it via
the `CudaTracer/Makefile`. 

To do this, place it at `CudaTracer/test/matrix_mult_smart.cu` and add `matrix_mult_smart`
to `TESTS_CU`. 

```Makefile
TESTS_CU += matrix_mult_smart
```

Then, compile and run.

```bash
make clean
make all
sudo ./matrix_mult_smart
```

Now, we see that using the smart allocator on its first run performs almost exactly the 
same as the pinning allocator, on both runs. The key difference is that the profile
run is configured to pin all pages, while the use run only choses to pin allocation C.

The output below excludes debuging statements.
```
Initializing in Profiling Mode.
Duration: 3141 ms
Checksum of the final result: inf

Initializing in Optimized Mode.
Duration: 3142 ms
Checksum of the final result: inf
```

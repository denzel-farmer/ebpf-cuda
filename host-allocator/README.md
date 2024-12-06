# cuda-allocator

How to run:

1. Compile CUDA kernel 

`nvcc -c dummy_kernel.cu -o dummy_kernel.o`

2. Compile benchmark application

`nvcc -g -O2 -o cuda_benchmark_application cuda_benchmark_application.cu     MemoryPools.cpp CustomAllocatorManager.cpp dummy_kernel.o     -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -ldl -Xcompiler -pthread`

3. Run in profile Mode

`./cuda_benchmark_application profile 10`

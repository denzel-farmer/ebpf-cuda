#include "CustomAllocatorManager.h"
#include "MemoryPools.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <functional>


#include <fstream>

__global__ void dummy_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1.0f;
    }
}

// Global file stream for CSV
std::ofstream csv_file;

// Open CSV file and write headers
void initialize_csv(const std::string& filename) {
    csv_file.open(filename, std::ios::out | std::ios::trunc);
    if (csv_file.is_open()) {
        csv_file << "AllocType,AllocSizeKB,RuntimeMS\n"; // Column headers
    } else {
        std::cerr << "Failed to open CSV file for writing." << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Close CSV file
void finalize_csv() {
    if (csv_file.is_open()) {
        csv_file.close();
    }
}


enum class AllocationType {
    Pinned,
    Unpinned,
    CustomOptim,
    CustomProfile
};

struct TestParameters {
    AllocationType alloc_type;
    size_t min_alloc_size;  // Minimum allocation size in bytes
    size_t max_alloc_size;  // Maximum allocation size in bytes
    size_t alloc_step;      // Step size for increasing allocation size
    size_t single_alloc_iters;
    size_t multi_alloc_iters;
    size_t multi_alloc_transfers;
};

#define CHECK_CUDA_ERROR(call)                                                           \
	do {                                                                             \
		cudaError_t err = call;                                                  \
		if (err != cudaSuccess) {                                                \
			std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
				  << __FILE__ << ":" << __LINE__ << std::endl;           \
			exit(EXIT_FAILURE);                                              \
		}                                                                        \
	} while (0)

float do_single_transfer(float *host_ptr, float *device_ptr, size_t size)
{
	CHECK_CUDA_ERROR(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice));
	float *device_data = static_cast<float *>(device_ptr);
	dummy_kernel<<<256, 256>>>(device_data, size / sizeof(float));
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost));
	return host_ptr[0];
}



// Define types for allocation and deallocation functions
using AllocateFunc = std::function<void*(size_t)>;
using DeallocateFunc = std::function<void(void*, size_t)>;

// Functions for different allocation types
void* allocate_pinned(size_t size) {
    float* ptr;
    CHECK_CUDA_ERROR(cudaMallocHost(&ptr, size));
    return ptr;
}

void deallocate_pinned(void* ptr, size_t) {
    CHECK_CUDA_ERROR(cudaFreeHost(ptr));
}

void* allocate_unpinned(size_t size) {
    return malloc(size);
}

void deallocate_unpinned(void* ptr, size_t) {
    free(ptr);
}

void* allocate_custom(size_t size) {
    return g_allocator_manager.allocate_memory(size);
}

void deallocate_custom(void* ptr, size_t size) {
    g_allocator_manager.deallocate_memory(ptr, size);
}

// constexpr size_t single_alloc_iters = 25;
// constexpr size_t multi_alloc_iters = 2;
// constexpr size_t multi_alloc_transfers = 175;
constexpr size_t single_alloc_iters = 25;
constexpr size_t multi_alloc_iters = 2;
constexpr size_t multi_alloc_transfers = 175;
void perform_test_demo_pinned(size_t alloc_size)
{
	// Allocate target memory on GPU
	float *device_ptr;
	float *host_ptr;
	double sum = 0.0;
	CHECK_CUDA_ERROR(cudaMalloc(&device_ptr, alloc_size));

	// Do 100 allocations, transfer each to GPU target once
    cudaDeviceSynchronize();
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < single_alloc_iters; ++i) {
		CHECK_CUDA_ERROR(cudaMallocHost(&host_ptr, alloc_size));

		host_ptr[0] = 1.0f;
		host_ptr[1] = 2.0f;

		sum += do_single_transfer(host_ptr, device_ptr, alloc_size);

		CHECK_CUDA_ERROR(cudaFreeHost(host_ptr));
	}

	// Do 5 allocations, transfer each to GPU target 1000 times
	for (int i = 0; i < multi_alloc_iters; ++i) {
		CHECK_CUDA_ERROR(cudaMallocHost(&host_ptr, alloc_size));

		host_ptr[0] = 1.0f;
		host_ptr[1] = 2.0f;

        for (int j = 0; j < multi_alloc_transfers; ++j) {
            sum += do_single_transfer(host_ptr, device_ptr, alloc_size);
        }

		CHECK_CUDA_ERROR(cudaFreeHost(host_ptr));
	}

    cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_time =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;

	// Free all allocations
    std::cout << "Sum: " << sum << std::endl;
}

void perform_test_demo_unpinned(size_t alloc_size)
{
	// Allocate target memory on GPU
	float *device_ptr;
	float *host_ptr;
	double sum = 0.0;
	CHECK_CUDA_ERROR(cudaMalloc(&device_ptr, alloc_size));

	// Do 100 allocations, transfer each to GPU target once
    cudaDeviceSynchronize();
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < single_alloc_iters; ++i) {
		host_ptr = static_cast<float *>(malloc(alloc_size));
		if (host_ptr == nullptr) {
			std::cerr << "Malloc failed" << std::endl;
			exit(EXIT_FAILURE);
		}
		host_ptr[0] = 1.0f;
		host_ptr[1] = 2.0f;


        sum += do_single_transfer(host_ptr, device_ptr, alloc_size);
 
        free(host_ptr);
	}

	// Do 5 allocations, transfer each to GPU target 1000 times
	for (int i = 0; i < multi_alloc_iters; ++i) {
		host_ptr = static_cast<float *>(malloc(alloc_size));
		if (host_ptr == nullptr) {
			std::cerr << "Malloc failed" << std::endl;
			exit(EXIT_FAILURE);
		}

		host_ptr[0] = 1.0f;
		host_ptr[1] = 2.0f;

        for (int j = 0; j < multi_alloc_transfers; ++j) {
            sum += do_single_transfer(host_ptr, device_ptr, alloc_size);
        }

        free(host_ptr);
	}


    cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_time =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;

	// Free all allocations
    std::cout << "Sum: " << sum << std::endl;
}

void perform_test_demo_smart(size_t alloc_size)
{
	// Allocate target memory on GPU
	float *device_ptr;
	float *host_ptr;
	double sum = 0.0;
	CHECK_CUDA_ERROR(cudaMalloc(&device_ptr, alloc_size));

	// Do 100 allocations, transfer each to GPU target once
    cudaDeviceSynchronize();
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < single_alloc_iters; ++i) {
		host_ptr = static_cast<float *>(g_allocator_manager.allocate_memory(alloc_size));
		if (host_ptr == nullptr) {
			std::cerr << "Malloc failed" << std::endl;
			exit(EXIT_FAILURE);
		}
		host_ptr[0] = 1.0f;
		host_ptr[1] = 2.0f;
        
        sum += do_single_transfer(host_ptr, device_ptr, alloc_size);

        g_allocator_manager.deallocate_memory(host_ptr, alloc_size);
	}

	// Do 5 allocations, transfer each to GPU target 1000 times
	for (int i = 0; i < multi_alloc_iters; ++i) {
		host_ptr = static_cast<float *>(g_allocator_manager.allocate_memory(alloc_size));
		if (host_ptr == nullptr) {
			std::cerr << "Malloc failed" << std::endl;
			exit(EXIT_FAILURE);
		}

		host_ptr[0] = 1.0f;
		host_ptr[1] = 2.0f;

        for (int j = 0; j < multi_alloc_transfers; ++j) {
            sum += do_single_transfer(host_ptr, device_ptr, alloc_size);
        }

        g_allocator_manager.deallocate_memory(host_ptr, alloc_size);
	}

    std::cout << "Total Amount Pinned: " << g_allocator_manager.total_amount_pinned << std::endl;

    std::cout << "Total Allocated: " << ((multi_alloc_iters + single_alloc_iters) * alloc_size) << std::endl;

    std::cout << "Total Percentage Pinned " << static_cast<double>(g_allocator_manager.total_amount_pinned)  / static_cast<double>((multi_alloc_iters + single_alloc_iters) * alloc_size) * 100 << "%" << std::endl; 

    cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_time =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;

	// Free all allocations
    std::cout << "Sum: " << sum << std::endl;
}

void perform_general_test(const TestParameters& params) {
    AllocateFunc allocate;
    DeallocateFunc deallocate;

    // Select allocation and deallocation functions
    switch (params.alloc_type) {
        case AllocationType::Pinned:
            allocate = allocate_pinned;
            deallocate = deallocate_pinned;
            break;
        case AllocationType::Unpinned:
            allocate = allocate_unpinned;
            deallocate = deallocate_unpinned;
            break;
        case AllocationType::CustomProfile:
            allocate = allocate_custom;
            deallocate = deallocate_custom;
            break;
        case AllocationType::CustomOptim:
            allocate = allocate_custom;
            deallocate = deallocate_custom;
            break;
        default:
            std::cerr << "Unknown Allocation Type" << std::endl;
            exit(EXIT_FAILURE);
    }

    for (size_t alloc_size = params.min_alloc_size; alloc_size <= params.max_alloc_size; alloc_size += params.alloc_step) {
        float* device_ptr;
        CHECK_CUDA_ERROR(cudaMalloc(&device_ptr, alloc_size));

        double sum = 0.0;
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        // Single allocation iterations
        for (size_t i = 0; i < params.single_alloc_iters; ++i) {
            float* host_ptr = static_cast<float*>(allocate(alloc_size));
            if (host_ptr == nullptr) {
                std::cerr << "Allocation failed" << std::endl;
                exit(EXIT_FAILURE);
            }

            host_ptr[0] = 1.0f;
            host_ptr[1] = 2.0f;
            sum += do_single_transfer(host_ptr, device_ptr, alloc_size);

            deallocate(host_ptr, alloc_size);
        }

        // Multiple allocation iterations with multiple transfers
        for (size_t i = 0; i < params.multi_alloc_iters; ++i) {
            float* host_ptr = static_cast<float*>(allocate(alloc_size));
            if (host_ptr == nullptr) {
                std::cerr << "Allocation failed" << std::endl;
                exit(EXIT_FAILURE);
            }

            host_ptr[0] = 1.0f;
            host_ptr[1] = 2.0f;

            for (size_t j = 0; j < params.multi_alloc_transfers; ++j) {
                sum += do_single_transfer(host_ptr, device_ptr, alloc_size);
            }

            deallocate(host_ptr, alloc_size);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        //std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;

        //std::cout << "Sum: " << sum << std::endl;

        // Log results to CSV
        if (csv_file.is_open()) {
            /*
            if ((params.alloc_type == AllocationType::Custom)) {
                // record total amount pinned
                csv_file << static_cast<int>(params.alloc_type) << ","
                     << (alloc_size / 1024) << "," // Convert to KB
                     << elapsed_time << "," 
                     << g_allocator_manager.total_amount_pinned << "," 
                     << (alloc_size * (params.multi_alloc_iters + params.single_alloc_iters)) << "," 
                     << (static_cast<double>(g_allocator_manager.total_amount_pinned)  / static_cast<double>(alloc_size * (params.multi_alloc_iters + params.single_alloc_iters))) * 100 
                     << "\n";
            }
            */
            
                csv_file << static_cast<int>(params.alloc_type) << ","
                     << (alloc_size / 1024) << "," // Convert to KB
                     << elapsed_time << "\n";
            
        }

        // Free device memory
        CHECK_CUDA_ERROR(cudaFree(device_ptr));
    }
}



int main(int argc, char *argv[]) {
	if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <single_iters> <multi_iters> <multi_transfers> <prepinned size>\n";
        return EXIT_FAILURE;
    }

    size_t single_iters = std::stoul(argv[1]);
    size_t multi_iters = std::stoul(argv[2]);
    size_t multi_transfers = std::stoul(argv[3]);
    size_t size = std::stoul(argv[4]);

    initialize_csv("benchmark_results.csv");

    // Define test parameters

    // single_iters, multi_iters, multi_transfers

    // pin 14 GB
    // writing one byte to every page
    //size_t size = 1024*1024*14;
    char* ptr;
    CHECK_CUDA_ERROR(cudaMallocHost(&ptr, size));
    for (int i = 0; i < size; i+=4096) {
        ptr[i] = 1;
    }


    std::vector<TestParameters> tests = {
        { AllocationType::Pinned, 1024*1024, 3*51200*1024, 16*1024*1024, single_iters, multi_iters, multi_transfers } // 1 KB to 50 MB in 1 KB steps
        //{ AllocationType::Unpinned, 1024*1024, 51200*1024, 8*1024*1024, single_iters, multi_iters, multi_transfers}
    };

    

    for (const auto& test : tests) {
        perform_general_test(test);
    }
    

    g_allocator_manager.initialize("profile");


    TestParameters profile_test = { AllocationType::CustomProfile, 1024*1024, 3*51200*1024, 16*1024*1024, single_iters, multi_iters, multi_transfers};
    perform_general_test(profile_test);


    finalize_csv();
  

    finalize_csv();

    return EXIT_SUCCESS;
}


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

__global__ void dummy_kernel(float *data, size_t size);

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

// constexpr size_t single_alloc_iters = 25;
// constexpr size_t multi_alloc_iters = 2;
// constexpr size_t multi_alloc_transfers = 175;
constexpr size_t single_alloc_iters = 1;
constexpr size_t multi_alloc_iters = 1;
constexpr size_t multi_alloc_transfers = 5;
void perform_test_demo_pinned(size_t alloc_size)
{
	// Allocate target memory on GPU
	float *device_ptr;
	float *host_ptr;
	double sum = 0.0;
	CHECK_CUDA_ERROR(cudaMalloc(&device_ptr, alloc_size));

	// Do 100 allocations, transfer each to GPU target once
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

	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_time =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;

	// Free all allocations
    std::cout << "Sum: " << sum << std::endl;
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <size_in_MB>\n";
		return EXIT_FAILURE;
	}
	std::cerr << "running main" << std::endl;
	size_t size_in_mb = std::stoul(argv[1]);
	size_t size_in_bytes = size_in_mb * 1024 * 1024;

    // std::cerr << "Running pinned memory test" << std::endl;
	// perform_test_demo_pinned(size_in_bytes);

    // // cuda sync
    // cudaDeviceSynchronize();

    // std::cerr << "Running unpinned memory test" << std::endl;
    // perform_test_demo_unpinned(size_in_bytes);

    std::cerr << "Running smart memory test" << std::endl;
    // Do profiling run
    g_allocator_manager.initialize("profile");
    perform_test_demo_smart(size_in_bytes);

	return EXIT_SUCCESS;
}

// void perform_test_basic(int num_allocations, size_t size, bool use_pinned) {
//     std::cout << "using basic" << std::endl;

//     std::vector<void*> host_ptrs(num_allocations, nullptr);
//     std::vector<void*> device_ptrs(num_allocations, nullptr);

//     for(int i = 0; i < 1; ++i) {
//         if(use_pinned) {
//             host_ptrs[i] = allocate_memory(size);
//             if(!host_ptrs[i]) {
//                 std::cerr << "Pinned allocation failed at index " << i << std::endl;
//                 exit(EXIT_FAILURE);
//             }
//         }
//         else {
//             host_ptrs[i] = malloc(size);
//             if(!host_ptrs[i]) {
//                 std::cerr << "Malloc failed at index " << i << std::endl;
//                 exit(EXIT_FAILURE);
//             }
//         }
//         memset(host_ptrs[i], 0, size);
//     }

//     for(int i = 0; i < 1; ++i) {
//         CHECK_CUDA_ERROR(cudaFree(device_ptrs[i]));
//         if(use_pinned) {
//             deallocate_memory(host_ptrs[i], size);
//         }
//         else {
//             free(host_ptrs[i]);
//         }
//     }
// }

// void perform_test(int num_allocations, size_t size, bool use_pinned,
//                  double& alloc_time, double& dealloc_time,
//                  double& h2d_time, double& kernel_time, double& d2h_time) {

//     std::vector<void*> host_ptrs(num_allocations, nullptr);
//     std::vector<void*> device_ptrs(num_allocations, nullptr);

//     int HOTSPOT = 2;

//     auto alloc_start = std::chrono::high_resolution_clock::now();
//     for(int i = 0; i < num_allocations; ++i) {
//         if(use_pinned) {
//             host_ptrs[i] = allocate_memory(size);
//             if(!host_ptrs[i]) {
//                 std::cerr << "Pinned allocation failed at index " << i << std::endl;
//                 exit(EXIT_FAILURE);
//             }
//         }
//         else {
//             host_ptrs[i] = malloc(size);
//             if(!host_ptrs[i]) {
//                 std::cerr << "Malloc failed at index " << i << std::endl;
//                 exit(EXIT_FAILURE);
//             }
//         }
//         memset(host_ptrs[i], 0, size);
//     }
//     auto alloc_end = std::chrono::high_resolution_clock::now();
//     alloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(alloc_end - alloc_start).count();

//     auto h2d_start = std::chrono::high_resolution_clock::now();
//     for(int i = 0; i < num_allocations; ++i) {
//         CHECK_CUDA_ERROR(cudaMalloc(&device_ptrs[i], size));
//         CHECK_CUDA_ERROR(cudaMemcpy(device_ptrs[i], host_ptrs[i], size, cudaMemcpyHostToDevice));
//     }
//     auto h2d_end = std::chrono::high_resolution_clock::now();
//     h2d_time = std::chrono::duration_cast<std::chrono::milliseconds>(h2d_end - h2d_start).count();

//     auto kernel_start = std::chrono::high_resolution_clock::now();
//     for(int i = 0; i < num_allocations; ++i) {
//         float* device_data = static_cast<float*>(device_ptrs[i]);
//         size_t num_elements = size / sizeof(float);
//         int threads = 256;
//         int blocks = (num_elements + threads - 1) / threads;
//         dummy_kernel<<<blocks, threads>>>(device_data, num_elements);
//         CHECK_CUDA_ERROR(cudaGetLastError());
//     }

//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//     auto kernel_end = std::chrono::high_resolution_clock::now();
//     kernel_time = std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end - kernel_start).count();

//     auto d2h_start = std::chrono::high_resolution_clock::now();
//     for(int i = 0; i < num_allocations; ++i) {
//         CHECK_CUDA_ERROR(cudaMemcpy(host_ptrs[i], device_ptrs[i], size, cudaMemcpyDeviceToHost));
//     }
//     auto d2h_end = std::chrono::high_resolution_clock::now();
//     d2h_time = std::chrono::duration_cast<std::chrono::milliseconds>(d2h_end - d2h_start).count();

//     auto dealloc_start = std::chrono::high_resolution_clock::now();
//     for(int i = 0; i < num_allocations; ++i) {
//         CHECK_CUDA_ERROR(cudaFree(device_ptrs[i]));
//         if(use_pinned) {
//             deallocate_memory(host_ptrs[i], size);
//         }
//         else {
//             free(host_ptrs[i]);
//         }
//     }
//     auto dealloc_end = std::chrono::high_resolution_clock::now();
//     dealloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(dealloc_end - dealloc_start).count();
// }

// 100 allocations, transfers to GPU once then frees, 100 unpinned
// 5 allocations pinned, transfers to GPU 1000 times each

// 100 allocations, transfers to GPU once then frees, 100 pinned
// 5 allocations pinned, transfers to GPU 1000 times each

// 100 allocations, transfers to GPU once then frees, 100 no
// // 5 allocations no, transfers to GPU 1000 times each

// void perform_test_simulated(size_t param){
//     std::cout << "running simulated performance test" << std::endl;

//     std::ofstream out_file("performance_test_results_2.csv", std::ios::app);

//     if (out_file.tellp() == 0) {
//         out_file << "data_size,first,second,runtime(ms)" << std::endl;  // Write the header only if the file is empty
//     }

//     std::vector<void*> host_ptrs(100, nullptr);
//     void* device_ptrs;

//     size_t size = static_cast<size_t>(1024 * param);
//     CHECK_CUDA_ERROR(cudaMalloc(&device_ptrs, size));

//     for (bool first : {true, false}) {
//         for (bool second : {true, false}) {
//             if (first == true && second == false){
//                 continue;
//             }

//             auto start = std::chrono::high_resolution_clock::now();
//             for (int i=0; i<100; i++){
//                 if (first == true){
//                     void* ptr = nullptr;
//                     cudaMallocHost(&ptr, size);
//                     host_ptrs[i] = ptr;
//                 } else{
//                     host_ptrs[i] = malloc(size);
//                 }

//                 CHECK_CUDA_ERROR(cudaMemcpy(device_ptrs, host_ptrs[i], size, cudaMemcpyHostToDevice));
//                 CHECK_CUDA_ERROR(cudaMemcpy(host_ptrs[i], device_ptrs, size, cudaMemcpyDeviceToHost));

//                 if (first == true){
//                     cudaFreeHost(host_ptrs[i]);
//                 }else{
//                     free(host_ptrs[i]);
//                 }
//             }

//             for (int i=0; i<5; i++){
//                 if (second == true){
//                     void* ptr = nullptr;
//                     cudaMallocHost(&ptr, size);
//                     host_ptrs[i] = ptr;
//                 } else{
//                     host_ptrs[i] = malloc(size);
//                 }

//                 for (int j=0; j<100; j++){
//                     CHECK_CUDA_ERROR(cudaMemcpy(device_ptrs, host_ptrs[i], size, cudaMemcpyHostToDevice));
//                     CHECK_CUDA_ERROR(cudaMemcpy(host_ptrs[i], device_ptrs, size, cudaMemcpyDeviceToHost));
//                 }

//                 if (second == true){
//                     cudaFreeHost(host_ptrs[i]);
//                 }else{
//                     free(host_ptrs[i]);
//                 }
//             }
//             auto end = std::chrono::high_resolution_clock::now();

//             long long runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

//             out_file << param << ","
//                     << (first ? "true" : "false") << ","
//                      << (second ? "true" : "false") << ","
//                      << runtime_ms << std::endl;
//         }
//     }
//     out_file.close();

// }

// void save_benchmark_results_to_csv(int num_iterations, double avg_alloc_pinned, double avg_dealloc_pinned,
//                                 double avg_h2d_pinned, double avg_kernel_pinned, double avg_d2h_pinned,
//                                 double avg_alloc_malloc, double avg_dealloc_malloc, double avg_h2d_malloc,
//                                 double avg_kernel_malloc, double avg_d2h_malloc, double data_size_gb,
//                                 const std::string& mode, int num_allocations) {

//     std::ofstream csvFile("benchmark_results.csv", std::ios::app);

//     if (!csvFile.is_open()) {
//         std::cerr << "Error opening file for writing.\n";
//         return;
//     }

//     if (csvFile.tellp() == 0) {
//         csvFile << "Num Iterations,Data Size (GB),Mode,Num Allocations,"
//                 << "Avg Alloc Time Pinned (s),Avg Dealloc Time Pinned (s),Avg Host to Device Time Pinned (s),"
//                 << "Avg Kernel Execution Time Pinned (s),Avg Device to Host Time Pinned (s),"
//                 << "Avg Alloc Time Malloc (s),Avg Dealloc Time Malloc (s),Avg Host to Device Time Malloc (s),"
//                 << "Avg Kernel Execution Time Malloc (s),Avg Device to Host Time Malloc (s)\n";
//     }

//     csvFile << num_iterations << "," << data_size_gb << "," << mode << "," << num_allocations << ","
//             << avg_alloc_pinned / 1e3 << "," << avg_dealloc_pinned / 1e3 << "," << avg_h2d_pinned / 1e3 << ","
//             << avg_kernel_pinned / 1e3 << "," << avg_d2h_pinned / 1e3 << ","
//             << avg_alloc_malloc / 1e3 << "," << avg_dealloc_malloc / 1e3 << "," << avg_h2d_malloc / 1e3 << ","
//             << avg_kernel_malloc / 1e3 << "," << avg_d2h_malloc / 1e3 << "\n";

//     csvFile.close();
// }

// int mainold(int argc, char* argv[]) {
//     if(argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <mode> <data_size_in_GB>\n";
//         std::cerr << "Modes:\n";
//         std::cerr << "  profile - Track allocation frequencies and save to file.\n";
//         std::cerr << "  use     - Load frequency data and optimize allocations.\n";
//         return EXIT_FAILURE;
//     }

//     std::string mode = argv[1];
//     if(mode != "profile" && mode != "use") {
//         std::cerr << "Invalid mode: " << mode << ". Use 'profile' or 'use'.\n";
//         return EXIT_FAILURE;
//     }

//     double data_size_gb = std::stod(argv[2]);
//     if(data_size_gb <= 0) {
//         std::cerr << "Data size must be positive.\n";
//         return EXIT_FAILURE;
//     }

//     size_t data_size_bytes = static_cast<size_t>(data_size_gb * 1024 * 1024 * 1024); // GB to bytes

//     // test parameters
//     int num_allocations = 10; // Number of GB-level allocations
//     // Adjust based on GPU memory capacity
//     // e.g. 10 allocations of 1 GB each = 10 GB total

//     // each allocation is data_size_bytes / num_allocations
//     size_t block_size = data_size_bytes / num_allocations;

//     int num_iterations = 3;

//     std::vector<double> alloc_times_pinned, dealloc_times_pinned;
//     std::vector<double> h2d_times_pinned, kernel_times_pinned, d2h_times_pinned;

//     std::vector<double> alloc_times_malloc, dealloc_times_malloc;
//     std::vector<double> h2d_times_malloc, kernel_times_malloc, d2h_times_malloc;

//     std::cout << "Starting Benchmark...\n";
//     std::cout << "Mode: " << mode << "\n";
//     std::cout << "Total Data Size: " << data_size_gb << " GB\n";
//     std::cout << "Number of Allocations: " << num_allocations << "\n";
//     std::cout << "Block Size per Allocation: " << block_size / (1024 * 1024) << " MB\n";
//     std::cout << "Number of Iterations: " << num_iterations << "\n\n";

//     g_allocator_manager.initialize(mode);

//     perform_test_simulated(data_size_gb);
//     std::cout << "stop here" << std::endl;
//     return 0;

//     for(int iter = 0; iter < num_iterations; ++iter) {
//         std::cout << "Iteration " << (iter + 1) << " / " << num_iterations << " - Pinned Memory\n";
//         double alloc_time, dealloc_time, h2d_time, kernel_time, d2h_time;
//         perform_test(num_allocations, block_size, true, alloc_time, dealloc_time, h2d_time, kernel_time, d2h_time);
//         alloc_times_pinned.push_back(alloc_time);
//         dealloc_times_pinned.push_back(dealloc_time);
//         h2d_times_pinned.push_back(h2d_time);
//         kernel_times_pinned.push_back(kernel_time);
//         d2h_times_pinned.push_back(d2h_time);

//         std::cout << "Iteration " << (iter + 1) << " / " << num_iterations << " - Malloc\n";
//         perform_test(num_allocations, block_size, false, alloc_time, dealloc_time, h2d_time, kernel_time, d2h_time);
//         alloc_times_malloc.push_back(alloc_time);
//         dealloc_times_malloc.push_back(dealloc_time);
//         h2d_times_malloc.push_back(h2d_time);
//         kernel_times_malloc.push_back(kernel_time);
//         d2h_times_malloc.push_back(d2h_time);
//     }

//     auto average = [](const std::vector<double>& times) -> double {
//         double sum = 0.0;
//         for(auto t : times) sum += t;
//         return sum / times.size();
//     };

//     double avg_alloc_pinned = average(alloc_times_pinned);
//     double avg_dealloc_pinned = average(dealloc_times_pinned);
//     double avg_h2d_pinned = average(h2d_times_pinned);
//     double avg_kernel_pinned = average(kernel_times_pinned);
//     double avg_d2h_pinned = average(d2h_times_pinned);

//     double avg_alloc_malloc = average(alloc_times_malloc);
//     double avg_dealloc_malloc = average(dealloc_times_malloc);
//     double avg_h2d_malloc = average(h2d_times_malloc);
//     double avg_kernel_malloc = average(kernel_times_malloc);
//     double avg_d2h_malloc = average(d2h_times_malloc);

//     std::cout << "\nBenchmark Results (Average over " << num_iterations << " iterations):\n\n";
//     std::cout << std::fixed << std::setprecision(2);
//     std::cout << "=== Pinned Memory ===\n";
//     std::cout << "Average Allocation Time:     " << avg_alloc_pinned / 1e3 << " seconds\n";
//     std::cout << "Average Deallocation Time:   " << avg_dealloc_pinned / 1e3 << " seconds\n";
//     std::cout << "Average Host to Device Time: " << avg_h2d_pinned / 1e3 << " seconds\n";
//     std::cout << "Average Kernel Execution Time:" << avg_kernel_pinned / 1e3 << " seconds\n";
//     std::cout << "Average Device to Host Time: " << avg_d2h_pinned / 1e3 << " seconds\n\n";

//     std::cout << "=== Malloc ===\n";
//     std::cout << "Average Allocation Time:     " << avg_alloc_malloc / 1e3 << " seconds\n";
//     std::cout << "Average Deallocation Time:   " << avg_dealloc_malloc / 1e3 << " seconds\n";
//     std::cout << "Average Host to Device Time: " << avg_h2d_malloc / 1e3 << " seconds\n";
//     std::cout << "Average Kernel Execution Time:" << avg_kernel_malloc / 1e3 << " seconds\n";
//     std::cout << "Average Device to Host Time: " << avg_d2h_malloc / 1e3 << " seconds\n";

//     if(mode == "profile") {
//         g_allocator_manager.save_frequency_data("frequency_data.txt");
//     }

//     save_benchmark_results_to_csv(num_iterations, avg_alloc_pinned, avg_dealloc_pinned, avg_h2d_pinned,
//                               avg_kernel_pinned, avg_d2h_pinned, avg_alloc_malloc, avg_dealloc_malloc,
//                               avg_h2d_malloc, avg_kernel_malloc, avg_d2h_malloc, data_size_gb, mode, num_allocations);

//     return EXIT_SUCCESS;
// }
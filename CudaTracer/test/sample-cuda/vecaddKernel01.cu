///
/// vecAddKernel01.cu
/// This Kernel adds two Vectors A and B in C on GPU
/// using coalesced memory access.
/// 

// Iter 0: 
// blocks[0]/threads[0] -> elems[0]
// blocks[0]/threads[1] -> elems[1]
// blocks[1]/threads[0] -> elems[block-start] = elems[blockIdx.x * blockDim.x]
// blocks[n_b]/thread[n_t] -> elems[blockstart + n_t]

// Iter i:
// blocks[n_b]/thread[n_t] -> elems[blockstart + n_t] + total_len * i = elems[blockIdx.x * blockDim.x + threadIdx.x + blockDim.x*gridDim.x]

// First thread of first block should access first element on first iteration
// 

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    // number of threads accessing together 
    int stride_size = (blockDim.x)*(gridDim.x);
    int blockStartIndex  = blockIdx.x * blockDim.x;
    int threadStartIndex = blockStartIndex + threadIdx.x;

    int thread_id = (blockIdx.x)*(blockDim.x) + (threadIdx.x);

    int i, curr_index;
    
    for (i=0; i < N; i++) {
        curr_index = threadStartIndex + i*stride_size;
        C[curr_index] = A[curr_index] + B[curr_index];
    }
    
}

///
/// matmultKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

#define FOOTPRINT_SIZE 32

// By default, use coalesced writing
//#define NO_COALESCED_WRITE

// TODO add support for footprint size of 64? Must split out separate 'warp size' or similar

#define NUM_THREADS (BLOCK_SIZE*BLOCK_SIZE)
#define NUM_TILE_ELEMS (FOOTPRINT_SIZE*FOOTPRINT_SIZE)
// For loading, we load rows (of size FOOTPRINT_SIZE) concurrently with NUM_THREADS threads 
#define LOAD_ROUNDS (NUM_TILE_ELEMS/NUM_THREADS)
// Number of tile rows loaded at once
#define CONCUR_ROWS (NUM_THREADS/FOOTPRINT_SIZE)

// The type Matrix is really a MATRIX DESCRIPTOR. 
// Matrices are stored in row major order:
//       M[row,col] = *(M.elements + row * M.stride + col)
//
// A sub matrix is not copied but allocated in the full matrix.
//
// This requires the stride of the full matrix to properly get to the
// next row of the sub matrix (a block).
//
// Stride is the width in bytes from one element of the larger matrix 
// to the element in the same column but one row down.



  // Block i,j computes on A/B/C i*FP_SIZE,j*FP_SIZE -> (i+1)*FP_SIZE-1,(j+1)*FP_SIZE-1
  
  // Example: inputs are 64x64, block size is 32x32 
  // Block 0,0 works on A/B/C [0,0] to [1*32-1,1*32-1] -> [0,0] to [31, 31]
  // Block 0,1 works on A/B/C [0*32,1*32] to [1*32-1, 2*32-1] -> [0,32] to [31, 63]

  // Within block 0,0 thread 0,0 computes result values C[0,0],C[0,1],C[1,0],C[1,1]
  // [0,0]: A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
  // [0,1]: 

  // Phase 1: 
  // - Block 0,0 loads A tile 0,0 and B tile 0,0 -> computes partial results for each thread 
  // - Block 0,1 loads A tile 0,0 and B tile 0,1 -> computes partial results for each thread 
  // ....
  // Phase 2:
  // - Block 0,0 loads A tile 0,1 and B tile 1,0 -> adds to partial result for each thread
  // - Block 0,1 loads A tile 0,1 and B tile 1,1 -> add to partial result for each thread
  // ...
  
  // Overall structure of tiles and blocks
  // Total of WIDTH/TILE_SIZE = WIDTH/BLOCK_SIZE phases
  // In phase n:
    // Block 0,0 loads A tile 0,n and B tile n,0
    // Block 0,1 loads A tile 0,n and B tile n,1
    // Block i,j loads A tile i,n and B tile n,j

  // Within block,phase given input tiles A_tile and B_tile 
    // Each thread loads one lement of tile, block threads should coalesce loading rows from A_tile and B_tile 
    // Split tile into 2x2 chunks, each thread works on a chunk

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_col = threadIdx.x;
  int thread_row = threadIdx.y;
  // Consec_id = thread_col + thread_row * BLOCK_SIZE
  int consec_id = thread_col + thread_row * BLOCK_SIZE;
  int tile_load_row = (consec_id / FOOTPRINT_SIZE);
  int tile_load_col = (consec_id % FOOTPRINT_SIZE);
  
  // This thread working on chunk thread_row,thread_col 
  // Starts at element thread_row*2, thread_col*2
  int tile_calc_row = thread_row*2;
  int tile_calc_col = thread_col*2;

  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  float Cvalue[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  // For each phase 
  for (int phase = 0; phase < (A.width / FOOTPRINT_SIZE); ++phase){
    // Get descriptors for Asub and Bsub
    // Tile i,j has corners [i*BLOCK_SIZE, j*BLOCK_SIZE] and [(i+1)*BLOCK_SIZE-1, (j+1)*BLOCK_SIZE-1]
    // Asub is tile (Block.x, phase) -> starts at [Block.x*BLOCK_SIZE, phase*BLOCK_SIZE]
    // Index into elements is elements + row*stride + col -> (Block.x*BLOCK_SIZE)*A.stride + phase*BLOCK_SIZE
    //
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * phase];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * phase + FOOTPRINT_SIZE * block_col];

    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Do a coalesced load for tile of A and B (load each 32-element row)
    // Load row i with threads 0 to FOOTPRINT_SIZE-1
    // Concurrently, can load NUM_THREADS / FOOTPRINT_SIZE rows 
    // Require (FOOTPRINT_SIZE/BLOCK_SIZE)^2 iterations
    for (int ld_round = 0; ld_round < LOAD_ROUNDS; ld_round++){ 
      // Load each row using FOOTPRINT_SIZE consecutive threads
      // Need to map thread coordinate (16x16) onto tile load coordinate (32xLOAD_ROUNDS)
      int round_load_row = ld_round*CONCUR_ROWS + tile_load_row;
      int global_tile_index = round_load_row*A.stride + tile_load_col;
      shared_A[round_load_row][tile_load_col] = Asub[global_tile_index];
      shared_B[round_load_row][tile_load_col] = Bsub[global_tile_index];
    }

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing 4 Cvalues by accumulation
    // TODO allow for variable chunk size?
#pragma unroll
    for(int e=0; e<FOOTPRINT_SIZE; ++e) {
      Cvalue[0][0] += shared_A[tile_calc_row][e] * shared_B[e][tile_calc_col];
      Cvalue[0][1] += shared_A[tile_calc_row][e] * shared_B[e][tile_calc_col + 1];
      Cvalue[1][0] += shared_A[tile_calc_row+1][e] * shared_B[e][tile_calc_col];
      Cvalue[1][1] += shared_A[tile_calc_row+1][e] * shared_B[e][tile_calc_col + 1];      
    }
    
    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }
  
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];


  #ifndef NO_COALESCED_WRITE
    // Option 1: coalesced--each thread writes its own value to shared memory, then writes coalesced to global memory
    __shared__ float shared_C[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // All threads write their values to shared memory
    shared_C[tile_calc_row][tile_calc_col] = Cvalue[0][0];
    shared_C[tile_calc_row][tile_calc_col+1] = Cvalue[0][1];
    shared_C[tile_calc_row+1][tile_calc_col] = Cvalue[1][0];
    shared_C[tile_calc_row+1][tile_calc_col+1] = Cvalue[1][1];

    __syncthreads();

    // All threads write coalesced to global memory
    // Require (FOOTPRINT_SIZE/BLOCK_SIZE)^2 iterations
    for (int ld_round = 0; ld_round < LOAD_ROUNDS; ld_round++){ 
      // Load each row using FOOTPRINT_SIZE consecutive threads
      // Need to map thread coordinate (16x16) onto tile load coordinate (32xLOAD_ROUNDS)
      int round_load_row = ld_round*CONCUR_ROWS + tile_load_row;
      int global_tile_index = round_load_row*A.stride + tile_load_col;
      Csub[global_tile_index] = shared_C[round_load_row][tile_load_col];
    }
  #else
    // Option 2: not coalesced--each thread writes its own 4 values directly to global memory 
    Csub[tile_calc_row * C.stride + tile_calc_col] = Cvalue[0][0];
    Csub[tile_calc_row * C.stride + (tile_calc_col + 1)] = Cvalue[0][1];
    Csub[(tile_calc_row + 1) * C.stride + tile_calc_col] = Cvalue[1][0];
    Csub[(tile_calc_row + 1) * C.stride + (tile_calc_col + 1)] = Cvalue[1][1];
  #endif

}

#define PPROG_IMPLEMENTATION

#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

#include <chrono>
#include <thread>
#include <fstream>
#include <array>

// ===================================================================================
// CONFIGURATION

// ~1M elements, keep this as it is
#define N (1 << 20)

// (1 << 20) / 1024 = 1024
#define BLOCK_SIZE 1024 

// ===================================================================================
// UTILITIES

#define cuda_check_error()                                                     \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

// ===================================================================================
// @@ IMPLEMENTATION

//@@ Sequential INCLUSIVE scan 
std::vector<int> scan_cpu(int* elements, int n) {
    
    std::vector<int> result(N);
    result[0] = elements[0];

    for (int i = 1; i < n; ++i)
        result[i] += result[i - 1] + elements[i];

    return result;
}

//@@ Write the INCLUSIVE scan fixup kernel for multi-block grids
__global__ void scan_fixup_gpu(int* result, int* scan_blocks, int n) {
  int block_start = 2 * BLOCK_SIZE * blockIdx.x;
  int tid = threadIdx.x;
  
  // Skip first block as it doesn't need fixup
  if (blockIdx.x == 0) return;
  
  // Get the prefix sum from previous block
  int prefix_sum = scan_blocks[blockIdx.x - 1];
  
  // Calculate indices for both segments of current block
  int idx1 = block_start + tid;
  int idx2 = block_start + tid + BLOCK_SIZE;

  // Add prefix sum to both segments of current block
  if (idx1 < n) result[idx1] += prefix_sum;
  if (idx2 < n) result[idx2] += prefix_sum;
}

//@@ Write the INCLUSIVE scan kernel using the up-down sweep algorithm
//   NOTE: 'blocks' can be NULL! 
__global__ void scan_gpu(int* elements, int* result, int* blocks, int n) {
  // Shared memory for storing block data (2 elements per thread)
  __shared__ int temp[2 * BLOCK_SIZE];
  
  // Calculate block starting index in global memory
  int block_start = 2 * BLOCK_SIZE * blockIdx.x;
  int tid = threadIdx.x;
  
  // Calculate global indices for both elements this thread handles
  int idx1 = block_start + tid;
  int idx2 = block_start + tid + BLOCK_SIZE;
  // Load two elements per thread from global to shared memory, use bounds checking to avoid out-of-range access
  temp[tid] = (idx1 < n) ? elements[idx1] : 0;
  temp[tid + BLOCK_SIZE] = (idx2 < n) ? elements[idx2] : 0;
  
  // Synchronize to ensure all data is loaded before processing
  __syncthreads();
  
  // Do scan sequentially using thread 0
  if (tid == 0) {
      for (int i = 1; i < 2 * BLOCK_SIZE; i++) {
          temp[i] += temp[i-1];
      }
  }
  
  // Synchronize to ensure scan is complete before other threads read results
  __syncthreads();
  
  // Store block sum for multi-block case, only thread 0 writes the block sum to avoid race conditions
  if (blocks != NULL && tid == 0) {
      blocks[blockIdx.x] = temp[2 * BLOCK_SIZE - 1];
  }
  
  // Write results from shared memory back to global memory, with bounds checking
  if (idx1 < n) result[idx1] = temp[tid];
  if (idx2 < n) result[idx2] = temp[tid + BLOCK_SIZE];
}

// ===================================================================================

int main(int argc, char** argv) {

    std::vector<int> elements = random_array(1, 100, N);

    std::vector<int> result_cpu;
    std::vector<int> result_gpu;
    result_gpu.resize(N);

    // Allocate matrix in GPU memory
    int *d_elements, *d_result;
    cudaMalloc(&d_elements, N * sizeof(int));
    cudaMemcpy(d_elements, elements.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_result, N * sizeof(int));
    cudaMemset(d_result, 0, N * sizeof(int));
    cuda_check_error();

    // Each block is responsible for (2 * BLOCK_SIZE) elements
    const int block_count = ceil((float)N / (BLOCK_SIZE * 2));
    printf("block_count: %d\n", block_count);

    // Create result and auxiliary arrays
    int *d_blocks, *d_scan_blocks;
    cudaMalloc(&d_blocks, 2 * block_count * sizeof(int));
    cudaMalloc(&d_scan_blocks, 2 * block_count * sizeof(int));
    cuda_check_error();

    // ===================================================================================
    // SEQUENTIAL

    auto timer_cpu = timerCPU{};
    timer_cpu.start();

    // Sequential
    result_cpu = scan_cpu(elements.data(), N);

    timer_cpu.stop();

    // ===================================================================================
    // GPU NAIVE

    auto timer_gpu = timerGPU{};
    timer_gpu.start();

    // Local scan inside each block, also stores block largest value
    // - d_elements contains the input values
    // - d_result will contain the partial scan of each block
    // - d_blocks will contain the largest/last value of each block
    scan_gpu<<<block_count, BLOCK_SIZE>>>(d_elements, d_result, d_blocks, N);
    cudaDeviceSynchronize();

    // Scan of all the blocks largest values
    // - d_blocks contains the largest/last value of each block
    // - d_scan_blocks will contain the scan of the blocks largest values
    scan_gpu<<<1, BLOCK_SIZE>>>(d_blocks, d_scan_blocks, NULL, BLOCK_SIZE * 2);
    cudaDeviceSynchronize();
    
    // Fixes local scan values (d_result) using the scan of all the blocks largest values
    // - d_result contains the partial scan of each block
    // - d_scan_block contains the scan of the blocks largest values
    scan_fixup_gpu<<<block_count, BLOCK_SIZE>>>(d_result, d_scan_blocks, N);
    cudaDeviceSynchronize();

    timer_gpu.stop();

    // Move result matrix to CPU memory
    cudaMemcpy(result_gpu.data(), d_result, N * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error();

    // ===================================================================================
    // TIMERS

    auto cpu_ms = timer_cpu.elapsed_ms();
    auto gpu_ms = timer_gpu.elapsed_ms();

    printf("reduction CPU\n\t%f ms\n", cpu_ms);

    printf("reduction GPU\n\t%f ms (speedup: cpu %.2fx)\n", 
        gpu_ms, cpu_ms / gpu_ms);

    // ===================================================================================
    // CHECK

    //for (int i = 0; i < N; ++i) {
    //    printf("%6d\t%6d\t%6d\n", elements[i], result_cpu[i], result_gpu[i]);
    //}

    bool ok_naive = (result_cpu == result_gpu);
    printf("Solution CPU vs GPU: %s\n", ok_naive ? "CORRECT" : "INCORRECT");

    // ===================================================================================

    return 0;
}
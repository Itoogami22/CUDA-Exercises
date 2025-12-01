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

// ~1B elements
#define N 1000000000

#define BLOCK_SIZE 512


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

//@@ Sequential reduction 
int reduce_cpu(int* elements, int n) {
    int result = 0;
    for (int i = 0; i < n; ++i)
        result += elements[i];
    return result;
}

//@@ Write the reduction kernel using local shared memory computation and a final atomic operation
__global__ void reduction_gpu(int* elements, int* result, int n) {
  //@@...
  __shared__ int SMem[BLOCK_SIZE];
  
  int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  SMem[threadIdx.x] = elements[GlobalIndex];
  
  /*
    if (GlobalIndex < n)
      SMem[threadIdx.x] = elements[GlobalIndex];
    else
      SMem[threadIdx.x] = 0;
  */

  __syncthreads();
  
  for (int i = 1; i < blockDim.x; i *= 2) {
    int index = threadIdx.x * i * 2;
      if (index < blockDim.x)
        SMem[index] += SMem[index + i];
      __syncthreads();
  }
  // Atomic addition to global result
  if (threadIdx.x == 0) {
    atomicAdd((unsigned int*)(result), SMem[0]);
  }
}


// ===================================================================================

int main(int argc, char** argv) {

    std::vector<int> elements = random_array(1, 100, N);

    int result_cpu;
    int result_gpu;

    // Allocate matrix in GPU memory
    int *d_elements;
    cudaMalloc((void**)&d_elements, N * sizeof(int));
    cudaMemcpy(d_elements, elements.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error();

    // Create result array
    int *d_reduction;
    cudaMalloc(&d_reduction, sizeof(int));
    cudaMemset(d_reduction, 0, sizeof(int));
    cuda_check_error();

    // ===================================================================================
    // SEQUENTIAL

    auto timer_cpu = timerCPU{};
    timer_cpu.start();

    // Sequential
    result_cpu = reduce_cpu(elements.data(), N);

    timer_cpu.stop();

    // ===================================================================================
    // GPU

    const int block_count = ceil((float)N / BLOCK_SIZE);
    printf("block_count: %d\n", block_count);

    auto timer_gpu = timerGPU{};
    timer_gpu.start();

    // Kernel launch
    reduction_gpu<<<block_count, BLOCK_SIZE>>>(d_elements, d_reduction, N);

    timer_gpu.stop();

    // Move result matrix to CPU memory
    cudaMemcpy(&result_gpu, d_reduction, sizeof(int), cudaMemcpyDeviceToHost);
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

    bool ok_naive = (result_cpu == result_gpu);
    printf("Solution CPU vs GPU: %s\n", ok_naive ? "CORRECT" : "INCORRECT");

    // ===================================================================================

    return 0;
}

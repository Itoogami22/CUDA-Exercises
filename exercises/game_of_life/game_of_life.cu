#define PPROG_IMPLEMENTATION

#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

#include <chrono>
#include <thread>

#include <raylib.h>

// ===================================================================================
// CONFIGURATION

#define TILE_SIZE_X 32
#define TILE_SIZE_Y 32

#define WIDTH  640
#define HEIGHT 360

// Set to 1 to use the shmem kernel
#define USE_SHMEM 0

// ===================================================================================
// UTILITIES

// Access the matrix by indices
#define at(i, j) (((i) * WIDTH + (j)))

// Get value in the matrix
#define matrix_get(matrix, i, j) ((at(i, j) <= WIDTH * HEIGHT - 1) ? matrix[at(i, j)] : 0)

// Check cuda errors
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

//@@ Write the stencil kernel without shared memory
__global__ void stencil_gpu_naive(int* current, int* next) {

    // Global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Bounds check to avoid out-of-bounds access
    if (x >= WIDTH || y >= HEIGHT)
        return;

    int idx = at(y, x); // Linear index in the grid
    int live_neighbors = 0; // Number of live neighbors

    // Count live neighbors 3x3 pattern exluding center cell
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0)
                continue; // Skip current cell

            int neighbor_x = x + dx;
            int neighbor_y = y + dy;
            
            // Check if neighbor is within bounds
            if (neighbor_x >= 0 && neighbor_x < WIDTH && neighbor_y >= 0 && neighbor_y < HEIGHT) {
                live_neighbors += current[at(neighbor_y, neighbor_x)];
            }
        }
    }

    // Apply Game of Life rules
    if (current[idx] == 1) {
        // Cell is currently alive
        if (live_neighbors < 2 || live_neighbors > 3) {
            next[idx] = 0; // Cell dies
        } else {
            next[idx] = 1; // Cell lives
        }
    } else {
        // Cell is currently dead
        if (live_neighbors == 3) {
            next[idx] = 1; // Cell becomes alive
        } else {
            next[idx] = 0; // Cell remains dead
        }
    }
}

//@@ Write the stencil kernel with shared memory
__global__ void stencil_gpu_shmem(int* current, int* next) {

    // Shared memory tile with halo regions
    __shared__ int shmem[TILE_SIZE_Y + 2][TILE_SIZE_X + 2];

    // Global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Local thread coordinates in shared memory
    int local_x = threadIdx.x + 1; // +1 for halo
    int local_y = threadIdx.y + 1; // +1 for halo

    // Load central tile data
    if (x < WIDTH && y < HEIGHT) {
        shmem[local_y][local_x] = current[at(y, x)];
    } else {
        shmem[local_y][local_x] = 0; // Out of bounds --> set zero (dead cell)
    }

    // LEFT halo (left most thread in block)
    if (threadIdx.x == 0 && x > 0) {
        shmem[local_y][0] = current[at(y, x - 1)];
    } 
    // Right halo (right most thread in block)
    else if (threadIdx.x == blockDim.x - 1 && x < WIDTH - 1) {  
        shmem[local_y][local_x + 1] = current[at(y, x + 1)];
    }
    // TOP halo (top most thread in block)
    if (threadIdx.y == 0 && y > 0) {
        shmem[0][local_x] = current[at(y - 1, x)];
    } 
    // BOTTOM halo (bottom most thread in block)
    else if (threadIdx.y == blockDim.y - 1 && y < HEIGHT - 1) {
        shmem[local_y + 1][local_x] = current[at(y + 1, x)];
    }

    // Synchronize threads to ensure all data is loaded
    __syncthreads();
    
    // Bounds check to avoid out-of-bounds access
    if (x >= WIDTH || y >= HEIGHT)
        return;

    // Linear index in the grid
    int idx = at(y, x);
    int live_neighbors = 0;

    // Count live neighbors using shared memory
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            // Skip current cell
            if (dx == 0 && dy == 0)
                continue;
            live_neighbors += shmem[local_y + dy][local_x + dx];
        }
    }
    // Apply Game of Life rules
    if (shmem[local_y][local_x] == 1) {
        // Cell is currently alive
        if (live_neighbors < 2 || live_neighbors > 3) {
            next[idx] = 0; // Cell dies
        } else {
            next[idx] = 1; // Cell lives
        }
    } else {
        // Cell is currently dead
        if (live_neighbors == 3) {
            next[idx] = 1; // Cell becomes alive
        } else {
            next[idx] = 0; // Cell remains dead
        }
    }
}

// ===================================================================================

void update_gpu(int* d_current, int* d_next, int* h_current) {

    dim3 block(TILE_SIZE_X, TILE_SIZE_Y);
    dim3 grid((WIDTH + TILE_SIZE_X - 1) / TILE_SIZE_X, (HEIGHT + TILE_SIZE_Y - 1) / TILE_SIZE_Y);

    auto frame_timer = timerGPU{};
    frame_timer.start();

#if USE_SHMEM == 0
    stencil_gpu_naive<<<grid, block>>>(d_current, d_next);
#endif

#if USE_SHMEM == 1
    stencil_gpu_shmem<<<grid, block>>>(d_current, d_next);
#endif

    frame_timer.stop();
    printf("frame time gpu: %f ms\n", frame_timer.elapsed_ms());

    // Store result back to CPU
    cudaMemcpy(h_current, d_next, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_current, d_next, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToDevice);
}

int main(int argc, char** argv) {

    // Generates random matrix data
    printf("generating random data...\n");
    auto h_current = random_matrix<int>(WIDTH, HEIGHT, 2043 /* SEED */);
    printf("done\n");

    // Allocate matrix in GPU memory
    int *d_current;
    cudaMalloc((void**)&d_current, h_current.size() * sizeof(int));
    cudaMemcpy(d_current, h_current.data(), h_current.size() * sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error();

    // Create result matrix
    int *d_next;
    cudaMalloc(&d_next, h_current.size() * sizeof(int));
    cuda_check_error();

    InitWindow(1280, 720, "Game of Life");
    SetTargetFPS(60);

    Image image = GenImageColor(WIDTH, HEIGHT, WHITE);
    Texture2D render_target = LoadTextureFromImage(image);

    while (!WindowShouldClose()) {

        // Update state
        update_gpu(d_current, d_next, h_current.data());

        // Update texture
        for (int i = 0; i < WIDTH * HEIGHT; ++i)
            ((Color*)image.data)[i] = h_current[i] ? WHITE : BLACK; 

        UpdateTexture(render_target, image.data);

        BeginDrawing();
        ClearBackground(BLACK);

        //DrawTexture(render_target, 0, 0, WHITE);
        DrawTextureEx(render_target, {0, 0}, 0.0f, 2.0f, WHITE);
		
		EndDrawing();
	}

    // Free cuda memory
    cudaFree(d_current);
    cudaFree(d_next);
    UnloadTexture(render_target);
    UnloadImage(image);

    CloseWindow();
    return 0;
}

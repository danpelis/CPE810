#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions provided by CUDA
#include "helper_cuda.h"


#define TILE_WIDTH 16


__global__ void matrixMultiplier(float* d_A, float* d_B, float* d_C, int j, int k, int l) {

    // Allocate shared memory space
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];
    
    // Set block and thread position variables
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // Detemine current C row and column
    int C_Row = block_y * TILE_WIDTH + thread_y;
    int C_Col = block_x * TILE_WIDTH + thread_x;

    float C_value = 0;
    // Determine number of phases required and start iteration
    int phase_count = ceil(k /(float)TILE_WIDTH);
    for (int phase = 0; phase < phase_count; ++phase) {
        if (C_Row < j && (phase * TILE_WIDTH + thread_x) < k) {
            // Load A value into shared mem
            A_shared[thread_y][thread_x] = d_A[C_Row * k + phase * TILE_WIDTH + thread_x];
        }
        else {
            A_shared[thread_y][thread_x] = 0.0;
        }

        if (C_Col < l && (phase * TILE_WIDTH + thread_y) < k) {
            // Load B value into shared mem
            B_shared[thread_y][thread_x] = d_B[(phase * TILE_WIDTH + thread_y) * l + C_Col];
        }
        else {
            B_shared[thread_y][thread_x] = 0.0;
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            // Multiple A and B values and add to current C value
            C_value += A_shared[thread_y][i] * B_shared[i][thread_x];
        }
        __syncthreads();
    }
    if (C_Row < j && C_Col < l) {
        // Write C values to global memory
        d_C[C_Row * l + C_Col] = C_value;
    }
}

// Function to perform matrix multiplication on the CPU
void hostMatrixMultiplier(float* A, float* B, float* C, unsigned int row_Dim_A, unsigned int col_Dim_A, unsigned int col_Dim_B) {
    for (int c_row = 0; c_row < row_Dim_A; c_row++) {

        for (int c_col = 0; c_col < col_Dim_B; c_col++) {
            C[c_row * col_Dim_B + c_col] = 0;
            
            for (int i = 0; i < col_Dim_A; i++) {
                C[c_row * col_Dim_B + c_col] += A[c_row * col_Dim_A + i] * B[i * col_Dim_B + c_col];
            }
        }
    }
}

// Function to compare C matrix from GPU and C matrix from CPU
// Used to validate GPU's results
int checkResults(float* h_C, float* C, int size_C) {
    for (int i = 0; i < size_C; i++) {
        if (h_C[i] != C[i]) {
            printf("\nMatrices not equal!\n");
            return(1);
        }
    }
    printf("\nMatrices equal\n");
    return (0);
}



int main(int argc, char** argv) {

    size_t optind;
    int row_Dim_A = 2, col_Dim_A = 2, col_Dim_B = 2;
    // Check for input matrix sizes
    for (optind = 1; optind < argc && argv[optind][0] == '-'; optind++) {
        if (argv[optind][1] == 'i') {
            row_Dim_A = atoi(argv[optind + 1]);
            col_Dim_A = atoi(argv[optind + 2]);
            col_Dim_B = atoi(argv[optind + 3]);
        }
    }

    // Create Matrix A and B
    unsigned int size_A = row_Dim_A * col_Dim_A;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)(malloc(mem_size_A));

    unsigned int size_B = col_Dim_A * col_Dim_B;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)(malloc(mem_size_B));

    int row;

    // Fill Matricies
    srand(time(NULL));
    int i = 1;
    for (row = 0; row < size_A; row++) {
        h_A[row] = rand();
        i++;
    }

    i = 1;
    for (row = 0; row < size_B; row++) {
        h_B[row] = rand();
        i++;
    }


    /*
    // Print Matricies

    printf("Matrix A:");
    for (int i = 0; i < size_A; i++) {
        if (i % col_Dim_A == 0) {
            printf("\n");
        }
        printf("%f\t", h_A[i]);
    }
    printf("\n\n");

    printf("Matrix B:");
    for (int i = 0; i < size_B; i++) {
        if (i % col_Dim_B == 0) {
            printf("\n");
        }
        printf("%f\t", h_B[i]);
    }
    printf("\n\n");
    */


    // Declare device variables
    float* d_A, * d_B, * d_C;
    unsigned int size_C = row_Dim_A * col_Dim_B;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*)(malloc(mem_size_C));;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));


    // Allocate memory on the device
    checkCudaErrors(cudaMalloc((void**)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

    // Copy input matricies to device memory
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    dim3 DimGrid(ceil(col_Dim_B / (float)TILE_WIDTH) + 1, ceil(row_Dim_A / (float)TILE_WIDTH) + 1, 1), DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    printf("\nGrid Dimensions %d, %d: \n", DimGrid.x, DimGrid.y);
    printf("Block Dimensions %d, %d: \n\n", DimBlock.x, DimBlock.y);

    // Record the start event
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaEventRecord(start, stream));
    
    int nIter = 150;
    for (int j = 0; j < nIter; j++) {
        matrixMultiplier << <DimGrid, DimBlock >> > (d_A, d_B, d_C, row_Dim_A, col_Dim_A, col_Dim_B);
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    float flopsPerMatrixMul = 2.0 * static_cast<float>(col_Dim_A) *
        static_cast<float>(row_Dim_A) *
        static_cast<float>(col_Dim_B);
    float gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
        (msecPerMatrixMul / 1000.0f);
    printf("\nGPU Done\n");
    printf(
        "Performance= %.2f GFlop/s\n Time= %.3f msec\n Size= %.0f Ops\n" \
        " WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        DimBlock.x * DimBlock.y);



    // Copy result from device memory
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    // Free device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    /*
    printf("Matrix C:");
    for (int i = 0; i < size_C; i++) {
        if (i % col_Dim_B == 0) {
            printf("\n");
        }
        printf("%f\t", h_C[i]);
    }
    */

    float* C = (float*)(malloc(mem_size_C));;
    hostMatrixMultiplier(h_A, h_B, C, row_Dim_A, col_Dim_A, col_Dim_B);
    printf("\nCPU Done\n");


    /*
    printf("\nMatrix C:");
    for (int i = 0; i < size_C; i++) {
        if (i % col_Dim_B == 0) {
            printf("\n");
        }
        printf("%f\t", C[i]);
    }
    */


    checkResults(h_C, C, size_C);

    free(h_A);
    free(h_B);
    free(h_C);
}
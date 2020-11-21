#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions provided by CUDA
#include "helper_cuda.h"
#include "helper_timer.h"

#define SECTION_SIZE 2048
using namespace std;


/*
	Brent_Kung Scan Algo
*/
__device__ void bk_scan(int* input, int* output, int input_size, int blockId) {

	__shared__ int partial[SECTION_SIZE];
	int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;


	if (i < input_size) {
		partial[threadIdx.x] = input[i];
	}
	if (i + blockDim.x < input_size) {
		partial[threadIdx.x + blockDim.x] = input[i + blockDim.x];
	}

	// Reduction Tree
	for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * 2 * stride - 1;
		if (index < SECTION_SIZE) {
			partial[index] += partial[index - stride];
		}
	}

	// Reverse Tree
	for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < SECTION_SIZE) {
			partial[index + stride] += partial[index];
		}
	}

	__syncthreads();
	if (i < input_size) {
		output[i] = partial[threadIdx.x];
	}
	if (i + blockDim.x < input_size) {
		output[i + blockDim.x] = partial[threadIdx.x + blockDim.x];
	}
}


/*
	Stream-based Scan Algo
*/
__global__ void scan(int* input, int* output, int input_size, volatile int* scan_value, int* flags, int* block_counter) {
	// Dynamic Block Index Assignment
	__shared__ int sblockId;
	if (threadIdx.x == 0) {
		sblockId = atomicAdd(block_counter, 1);
	}
	__syncthreads();
	const int blockId = sblockId;


	// Step 1: Brent_Kung scan algo

	bk_scan(input, output, input_size, blockId);
	
	
	// Step 2: Get sum from block i - 1, Generate Sum and pass sum to block i + 1

	// Adjacent Synchronization:
	// Atomic operations on flags array and reads to scan_value array mostly happen in L2 caches
	__shared__ int previous_sum;
	if (threadIdx.x == 0) {
		// Wait for previous sum to be recorded
		while (atomicAdd(&flags[blockId], 0) == 0) {}

		// Read previous partial sum
		previous_sum = scan_value[blockId];

		// Record local partial sum
		scan_value[blockId + 1] = previous_sum + output[(blockId * SECTION_SIZE) + (SECTION_SIZE - 1)];
		//scan_value[blockId + 1] = previous_sum + partial[SECTION_SIZE - 1];

		// Memory Fence
		__threadfence();

		// Inform next block to continue by setting flag
		atomicAdd(&flags[blockId + 1], 1);

	}
	__syncthreads();



	// Step 3: Add sum received to all values in section

	int j = 2 * blockId * blockDim.x + threadIdx.x;
	if (j < input_size) {
		output[j] += previous_sum;
	}
	if (j + blockDim.x < input_size) {
		output[j + blockDim.x] += previous_sum;
	}
}


void cpu_scan(int* in_vec, int* h_cpu_out_vec, int vec_size) {
	for (int i = 0; i < vec_size; i++) {
		if (i > 0) {
			h_cpu_out_vec[i] = in_vec[i] + h_cpu_out_vec[i - 1];
		}
		else {
			h_cpu_out_vec[i] = in_vec[i];
		}
	}
}


bool check_results(int* h_out_vec, int* h_cpu_out_vec, int vec_size) {
	for (int i = 0; i < vec_size; i++) {
		if (h_out_vec[i] != h_cpu_out_vec[i]) {
			return false;
		}
	}
	return true;
}


int main(int argc, char** argv) {

	size_t optind;
	int input_vec_size = 2049;
	bool debug = false;
	for (optind = 1; optind < argc; optind++) {
		if (argv[optind][1] == 'i') {
			input_vec_size = atoi(argv[optind + 1]);
		}
		if (argv[optind][1] == 'd') {
			debug = true;
		}
	}

	const int vec_size = input_vec_size;
	unsigned int mem_vec_size = sizeof(int) * vec_size;
	int* h_in_vec = (int*)(malloc(mem_vec_size));
	int* h_out_vec = (int*)(malloc(mem_vec_size));
	int* h_cpu_out_vec = (int*)(malloc(mem_vec_size));

	// Fill Input Vector
	srand(time(NULL));
	int index;
	for (index = 0; index < vec_size; index++) {
		h_in_vec[index] = rand();
		//h_in_vec[index] = 1;
	}

	
	if (debug) {
		printf("\nInput Vec:\n");
		for (index = 0; index < vec_size; index++) {
			printf("%d\t", h_in_vec[index]);
		}
		printf("\n");
	}
	


	int* d_in_vec;
	int* d_out_vec;
 
	checkCudaErrors(cudaMalloc((void**)&d_in_vec, mem_vec_size));
	checkCudaErrors(cudaMalloc((void**)&d_out_vec, mem_vec_size));

	checkCudaErrors(cudaMemcpy(d_in_vec, h_in_vec, mem_vec_size, cudaMemcpyHostToDevice));

	int num_block = ceil((float)vec_size / SECTION_SIZE);
	dim3 DimGrid(num_block), DimBlock(SECTION_SIZE/2);
	printf("Grid Size: %d\nBlock Size: %d\n", num_block, SECTION_SIZE / 2);


	int* d_block_counter;
	int* d_flags;
	volatile int* scan_value;

	unsigned int mem_block_num = sizeof(int) * num_block;
	checkCudaErrors(cudaMalloc((void**)&d_block_counter, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_flags, mem_block_num));
	checkCudaErrors(cudaMalloc((void**)&scan_value, mem_block_num));

	
	int* h_flags = (int*)(calloc(num_block, sizeof(int)));
	h_flags[0] = 1;
	
	checkCudaErrors(cudaMemcpy(d_flags, h_flags, mem_block_num, cudaMemcpyHostToDevice));
	
	
	// Record the start event
	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	checkCudaErrors(cudaEventRecord(start, stream));


	printf("\nPerforming GPU List Scan...\n");
		
	scan<<<DimGrid, DimBlock>>>(d_in_vec, d_out_vec, vec_size, scan_value, d_flags, d_block_counter);

	printf("GPU Complete\n");

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	
	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerConv = msecTotal / 1.0f;
	float flopsPerBKScan = (2.0 * SECTION_SIZE - 2.0 - (log(SECTION_SIZE) / log(2))) * num_block;
	float flopsPerStreamScan = vec_size + num_block;
	float flopsPerConv = flopsPerBKScan + flopsPerStreamScan;
	float gigaFlops = (flopsPerConv * 1.0e-9f) / (msecPerConv / 1000.0f);
	printf("Performance= %.2f GFlop/s\n Time= %f msec\n Size= %.0f Ops\n", gigaFlops, msecPerConv, flopsPerConv);
	

	checkCudaErrors(cudaMemcpy(h_out_vec, d_out_vec, mem_vec_size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_in_vec));
	checkCudaErrors(cudaFree(d_out_vec));
	checkCudaErrors(cudaFree(d_block_counter));
	checkCudaErrors(cudaFree(d_flags));

	
	if (debug) {
		printf("\nGPU Results:\n");
		for (index = 0; index < vec_size; index++) {
			printf("%d\t", h_out_vec[index]);
		}
	}
	


	/*
		CPU Convolution
	*/
	printf("\nPerforming CPU List Scan...\n");

	cpu_scan(h_in_vec, h_cpu_out_vec, vec_size);

	printf("CPU Complete\n");
	if (debug) {
		printf("\nCPU Results:\n");
		for (index = 0; index < vec_size; index++) {
			printf("%d\t", h_cpu_out_vec[index]);
		}
	}
	
	/*
		Validation
	*/
	printf("\nChecking...\n");
	bool check = check_results(h_out_vec, h_cpu_out_vec, vec_size);
	if (check) {
		printf("Scan Results Equal\n");
	}
	else {
		printf("ERROR: Scan Results Differ\n");
	}



	free(h_in_vec);
	free(h_out_vec);
	free(h_cpu_out_vec);
}
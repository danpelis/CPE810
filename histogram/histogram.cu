#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions provided by CUDA
#include "helper_cuda.h"


using namespace std;

// Histogram calculated by the CPU
void sequential_Histogram(int data[], int histo[], int length, int bin_size) {

	for (int i = 0; i < length; i++) {
		int histo_position = floor(data[i] / bin_size);
		if (data[i] >= 0 && data[i] <= 1024) {
			histo[histo_position]++;
		}
	}
}


// Histogram calculted by the GPU using Strategy I
// Strat I divides array into blocks and has each thread
// work on a block
__global__ void histogramRoutineI(int d_data[], int d_histo[], int length, int bin_size) {
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int section_len = (length - 1) / (blockDim.x * gridDim.x) + 1;
	int start = i * section_len;

	for (int k = 0; k < section_len; k++) {
		if (start + k < length) {
			int histo_position = ceil(d_data[start + k] / (float) bin_size) - 1;
			if (d_data[start + k] >= 0 && d_data[start + k] <= 1024) {
				atomicAdd(&d_histo[histo_position],1);
			}
		}
	}
}

// Histogram calculted by the GPU using Strategy II
// Strat II assigns elements to threads with an offset
// of the total amount of threads
__global__ void histogramRoutineII(int d_data[], int d_histo[], int length, int bin_size, unsigned int num_bins) {
	
	unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	
	extern __shared__ unsigned int histo_s[];
	
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
		histo_s[binIdx] = 0u;
	}
	
	__syncthreads();
	
	unsigned int current_index;
	unsigned int previous_index = -1;
	unsigned int accumulator = 0;
	
	for (int i = thread_id; i < length; i += blockDim.x * gridDim.x) {
		int current_val = d_data[i];
		int histo_position = floor(current_val / (float)bin_size);
		
		if (current_val >= 0 && current_val <= 1024) {
			current_index = histo_position;
			if (current_index != previous_index) {
				if (accumulator > 0) {
					atomicAdd(&histo_s[previous_index], accumulator);
				}
				accumulator = 1;
				previous_index = current_index;
			}
			else {
				accumulator++;
			}
		}
	}
	
	if (accumulator > 0) {
		atomicAdd(&histo_s[current_index], accumulator);
	}

	
	__syncthreads();

	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
		atomicAdd(&(d_histo[binIdx]), histo_s[binIdx]);
	}
}

bool is_power_of_2(int x) {
	return x <= 256 && x > 2 && !(x & (x - 1));
}


int main(int argc, char** argv) {

	size_t optind;
	int input_vec_size = 1000, input_bin_num = 64;
	int grid_dim = 0, block_dim = 0;
	// Check for input number of bins and vector size
	for (optind = 1; optind < argc; optind++) {
		if (argv[optind][1] == 'i') {
			input_bin_num = atoi(argv[optind + 1]);
			input_vec_size = atoi(argv[optind + 2]);
		}

		if (argv[optind][1] == 'd') {
			grid_dim = atoi(argv[optind + 1]);
			block_dim = atoi(argv[optind + 2]);
		}

	}

	if (!is_power_of_2(input_bin_num)) {
		printf("Number of bins not in range defaulting to 4 bins.");
		input_bin_num = 64;
	}

	const int vec_size = input_vec_size;
	const int bin_num = input_bin_num;
	printf("Number of Bins:\n\t- %d\n", bin_num);
	printf("Input Vector Size:\n\t- %d\n", vec_size);


	// Calculate the size of bins 
	int bin_size = 1024 / bin_num;

	// Determine size of Data array and Histogram array
	unsigned int mem_size_data = sizeof(int) * vec_size;
	unsigned int mem_size_histo = sizeof(int) * bin_num;

	// Allocate memory for Data array and Histogram arrays
	int* h_data = (int*)(calloc(vec_size, sizeof(int)));
	int* h_histo = (int*)(calloc(bin_num, sizeof(int)));
	int* seq_histo = (int*)(calloc(bin_num, sizeof(int)));

	// Fill Data array with random values from 0 to 1024
	srand(time(NULL));
	int i;
	for (i = 0; i < vec_size; i++) {
		h_data[i] = rand() % 1024;
	}

	// Print values in the Data array
	//for (int i = 0; i < vec_size; ++i) {
	//	printf("%d\n", h_data[i]);
	//}

	// Allocate memory on the device for data and histogram arrays
	int* d_data;
	int* d_histo;
	checkCudaErrors(cudaMalloc((void**)&d_data, mem_size_data));
	checkCudaErrors(cudaMalloc((void**)&d_histo, mem_size_histo));

	// Copy Data array to device memory
	checkCudaErrors(cudaMemcpy(d_data, h_data, mem_size_data, cudaMemcpyHostToDevice));

	

	// Set Grid and Block dimensions
	if (grid_dim == 0) {
		grid_dim = (vec_size / 32) + 1;
	}
	if (block_dim == 0) {
		block_dim = 32;
	}
	
	dim3 DimGrid(grid_dim), DimBlock(block_dim);
	printf("\nGrid Dimensions:\n\t- %d \n", DimGrid.x);
	printf("Block Dimensions:\n\t- %d \n\n", DimBlock.x);

	// Record the start event
	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	checkCudaErrors(cudaEventRecord(start, stream));

	int iterations = 150;
	for (int j = 0; j < iterations; j++) {
		//histogramRoutineI << <DimGrid, DimBlock, sizeof(int) * bin_num >> > (d_data, d_histo, vec_size, bin_size);
		histogramRoutineII << <DimGrid, DimBlock, sizeof(int)* bin_num >> > (d_data, d_histo, vec_size, bin_size, bin_num);
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerHisto = msecTotal / iterations;
	float flopsPerHisto = 1.0 * vec_size + bin_num * (int)DimGrid.x;
	float gigaFlops = (flopsPerHisto * 1.0e-9f) /
		(msecPerHisto / 1000.0f);
	printf("\nGPU Done\n");
	printf(
		"Performance= %.2f GFlop/s\n Time= %.3f msec\n Size= %.0f Ops\n" \
		" WorkgroupSize= %u threads/block\n",
		gigaFlops,
		msecPerHisto,
		flopsPerHisto,
		DimBlock.x * DimBlock.y);

	
	// Copy result from device memory
	checkCudaErrors(cudaMemcpy(h_histo, d_histo, mem_size_histo, cudaMemcpyDeviceToHost));
	
	// Free device memory
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_histo));
	
	
	for (int i = 0; i < bin_num; i++) {
		h_histo[i] = h_histo[i] / iterations;
		//printf("Bin %d: %d\n", i, h_histo[i]);
	}
	

	
	sequential_Histogram(h_data, seq_histo, vec_size, bin_size);

	/*
	for (int i = 0; i < bin_num; ++i) {
		printf("%d\n", seq_histo[i]);
	}
	*/

	printf("Running Checks:\n");
	unsigned int histo_total = 0;
	for (int i = 0; i < bin_num; ++i) {
		histo_total += h_histo[i];
		if (seq_histo[i] != h_histo[i]) {
			printf("\t- ERROR: Histograms not equal\n");
		}
	}
	if (histo_total != vec_size) {
		printf("\t- ERROR: Histogram Missing Values!\n");
		printf("%u", histo_total);
	}
	else {
		printf("\t- All values counted\n");
	}
	printf("\t- Histograms Equal\n");

	free(seq_histo);
	free(h_histo);
	free(h_data);
}
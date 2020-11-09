#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions provided by CUDA
#include "helper_cuda.h"
#include "helper_timer.h"


#define TILE_WIDTH 32
__constant__ int d_mask[128];

using namespace std;


/*
	Method 1:
	These functions rely on blockDim.x being equal to dim_X.
	They then apply one thread to every element in the input image
*/
__global__ void convolution2D_X(int* d_img, int* d_out_img, int mask_dim, int img_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int ds_img[TILE_WIDTH];

	// Loads in block's private tile
	ds_img[threadIdx.x] = d_img[index];

	__syncthreads();

	int current_tile_start_point = blockIdx.x * blockDim.x;
	int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
	int img_start_point = index - (mask_dim / 2);

	int out_val = 0;
	for (int j = 0; j < mask_dim; j++) {
		int img_index = img_start_point + j;
		if (img_index >= 0 && img_index < img_size) {
			if ((img_index >= current_tile_start_point) && (img_index < next_tile_start_point)) {
				out_val += ds_img[threadIdx.x + j - (mask_dim / 2)] * d_mask[j];
			}
		}
	}
	d_out_img[index] = out_val;
}

__global__ void convolution2D_Y(int *d_img, int *d_out_img, int mask_dim, int img_size) {	
	int index_y = threadIdx.x * gridDim.x + blockIdx.x;
	__shared__ int ds_img[TILE_WIDTH];

	if (index_y >= 0 && index_y < img_size){
		// Loads in block's private tile
		ds_img[threadIdx.x] = d_img[index_y];
		__syncthreads();

		int out_val = 0;
		for (int j = 0; j < mask_dim; j++) {
			int shared_index = (threadIdx.x - (mask_dim / 2)) + j;
			if ((shared_index >= 0) && (shared_index < blockDim.x)) {
				out_val += ds_img[shared_index] * d_mask[j];
			}
		}
		d_out_img[index_y] = out_val;
	}
}



/*
	Method 2:
	These functions implement a sliding tile technique, where each thread block is organized into a tile
	that slides along the row (or column for Y function) to calculate all the elements.
	In this method each row (or column for Y function) is assigned one block.
*/
__global__ void convolution2D_X_2(int* d_img, int* d_out_img, int mask_dim, int img_size, const int dim_x) {

	__shared__ int ds_img[TILE_WIDTH];
	
	int row = blockIdx.x;
	int phases = (dim_x / TILE_WIDTH) + 1;
	int row_start = row * dim_x;
	int row_end = ((row + 1) * dim_x) - 1;
	
	for (int phase = 0; phase < phases; phase++) {

		// Loads in block's private tile
		int index = blockIdx.x * dim_x + TILE_WIDTH * phase + threadIdx.x;
		ds_img[threadIdx.x] = d_img[index];

		__syncthreads();

		int mask_radius = (mask_dim / 2);
		int out_val = 0;
		if (index >= row_start && index <= row_end) {
			for (int j = -mask_radius; j <= mask_radius; j++) {
				int tile_index = threadIdx.x - j;

				if (index >= 0 && index < img_size && index - j >= row_start && index - j <= row_end) {
					if (tile_index >= 0 && tile_index < TILE_WIDTH) {
						out_val += ds_img[tile_index] * d_mask[mask_radius - j];
					}
					else if (index - j >= row_start && index - j <= row_end) {
						out_val += d_img[index - j] * d_mask[mask_radius - j];
					}
				}
			}
			d_out_img[index] = out_val;
		}
	}
}

__global__ void convolution2D_Y_2(int* d_img, int* d_out_img, int mask_dim, int img_size, int dim_x, int dim_y) {
	__shared__ int ds_img[TILE_WIDTH];

	int col = blockIdx.x;
	int phases = (dim_y / TILE_WIDTH) + 1;
	int col_start = col;
	int col_end = col + dim_y * (dim_x);

	for (int phase = 0; phase < phases; phase++) {
		// Loads in block's private tile
		int index = col + (dim_x * threadIdx.x) + (TILE_WIDTH * phase * dim_x);
		
		ds_img[threadIdx.x] = d_img[index];

		__syncthreads();

		int mask_radius = (mask_dim / 2);
		int out_val = 0;

		if (index >= col_start && index <= col_end) {

			for (int j = -mask_radius; j <= mask_radius; j++) {
				int tile_index = threadIdx.x - j;

				if (index - (j * dim_x) >= 0 && index - (j * dim_x) < img_size && index - j >= col_start && index - j <= col_end) {

					if (tile_index >= 0 && tile_index < TILE_WIDTH) {
						out_val += ds_img[tile_index] * d_mask[mask_radius - j];
					}
					else if (index - j >= col_start && index - j <= col_end) {
						out_val += d_img[index - (j * dim_x)] * d_mask[mask_radius - j];
					}
				}
			}
			d_out_img[index] = out_val;
		}
	}
}



/*
	CPU Implementation:
	These functions perform the convolution on the CPU to validate the GPU's results.
*/
void convolution_CPU_X(int* h_img, int* h_cpu_out_img, int* h_mask, int mask_dim, int dim_x, int dim_y) {
	
	int current_val = 0;
	int current_index = 0;
	int new_val = 0;
	int start_point = 0;

	// Loop through num of rows
	for (int k = 0; k < dim_y; k++) {
		int row_start = current_index;

		// Loop through num of cols
		for (int i = 0; i < dim_x; i++) {
			new_val = 0;
			current_val = h_img[current_index];
			start_point = current_index - (mask_dim / 2);

			//Loop through mask
			for (int j = 0; j < mask_dim; j++) {
			
				if (start_point + j >= row_start && start_point + j < row_start + dim_x) {
					new_val += h_img[start_point + j] * h_mask[j];
				}
			}
			h_cpu_out_img[current_index] = new_val;
			current_index += 1;
		}
	}

}

void convolution_CPU_Y(int* h_img, int* h_cpu_out_img, int* h_mask, int mask_dim, int dim_x, int dim_y) {

	int current_val = 0;
	int current_index = 0;
	int new_val = 0;
	int start_point = 0;
	int end_point = 0;
	int col_start = 0;
	int col_end = 0;

	// Loop through num of cols
	for (int k = 0; k < dim_x; k++) {
		current_index = k;
		col_start = current_index;
		col_end = dim_x * (dim_y - 1) + current_index;

		// Loop through num of rows
		for (int i = 0; i < dim_y; i++) {
			new_val = 0;
			current_val = h_img[current_index];
			start_point = current_index - ((mask_dim / 2) * dim_x);
			end_point = current_index + ((mask_dim / 2) * dim_x);

			// Loop through mask
			int mask_index = 0;
			for (int j = 0; j < mask_dim; j ++) {
				if (start_point + mask_index >= col_start && start_point + mask_index <= col_end) {
					new_val += h_img[start_point + mask_index] * h_mask[j];
				}
				mask_index += dim_x;
			}
			h_cpu_out_img[current_index] = new_val;
			current_index += dim_x;
		}
	}
}


bool checkOutImgs(int* h_gpu_out_img, int* h_cpu_out_img, int img_size) {
	for (int i = 0; i < img_size; i++){
		if (h_gpu_out_img[i] != h_cpu_out_img[i]) {
			return false;
		}
	}
	return true;
}


int main(int argc, char** argv) {

	size_t optind;
	int input_dim_X = 11, input_dim_Y = 10, input_mask_dim = 3;
	bool debug = false;
	bool no_check = false;
	for (optind = 1; optind < argc; optind++) {
		if (argv[optind][1] == 'i') {
			input_dim_X = atoi(argv[optind + 1]);
		}
		if (argv[optind][1] == 'j') {
			input_dim_Y = atoi(argv[optind + 1]);
		}
		if (argv[optind][1] == 'k') {
			input_mask_dim = atoi(argv[optind + 1]);
		}
		if (argv[optind][1] == 'd') {
			debug = true;
		}
		if (argv[optind][1] == 'c') {
			no_check = true;
		}
	}

	const int dim_X = input_dim_X, dim_Y = input_dim_Y, mask_dim = input_mask_dim;
	unsigned int img_size = dim_X * dim_Y;
	unsigned int mem_img_size = sizeof(int) * img_size;
	int* h_img = (int*)(malloc(mem_img_size));
	int* h_out_img = (int*)(malloc(mem_img_size));
	int* h_cpu_out_img_x = (int*)(malloc(mem_img_size));
	int* h_cpu_out_img_y = (int*)(malloc(mem_img_size));

	// Fill Image Matrix
	srand(time(NULL));
	int row;
	for (row = 0; row < img_size; row++) {
		h_img[row] = rand() % 15;
	}

	if (debug) {
		printf("\nInput Image:\n");
		for (row = 0; row < img_size; row++) {
			if (row > 0 && row % dim_X == 0) {
				printf("\n");
			}
			printf("%d\t", h_img[row]);
		}
		printf("\n");
	}

	unsigned int mem_mask_size = sizeof(int) * mask_dim;
	int* h_mask = (int*)(calloc(mask_dim, sizeof(int)));

	for (int i = 0; i < mask_dim; i++) {
		h_mask[i] = rand() % 15;
	}

	printf("\nMask:\n");
	for (row = 0; row < mask_dim; row++) {
		printf("%d\t", h_mask[row]);
	}
	printf("\n");

	int* d_img;
	int* d_out_img_x;
	int* d_out_img_y;
	checkCudaErrors(cudaMalloc((void**)&d_img, mem_img_size));
	checkCudaErrors(cudaMalloc((void**)&d_out_img_x, mem_img_size));
	checkCudaErrors(cudaMalloc((void**)&d_out_img_y, mem_img_size));

	checkCudaErrors(cudaMemcpy(d_img, h_img, mem_img_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_mask, h_mask, mask_dim * sizeof(int)));

	// Method 1: Grid and Block
	// dim3 DimGrid_X(dim_Y), DimBlock_X(dim_X);
	// dim3 DimGrid_Y(dim_X), DimBlock_Y(dim_Y);

	// Method 2: Grid and Block
	dim3 DimGrid_X_Tiled(dim_Y), DimBlock_X_Tiled(TILE_WIDTH);
	dim3 DimGrid_Y_Tiled(dim_X), DimBlock_Y_Tiled(TILE_WIDTH);

	// Record the start event
	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	checkCudaErrors(cudaEventRecord(start, stream));


	printf("\nPerforming GPU Convolution...\n");

	int iterations = 150;
	for (int j = 0; j < iterations; j++) {

		// Method 1: Kernel Functions
		//convolution2D_X << <DimGrid_X, DimBlock_X >> > (d_img, d_out_img_x, mask_dim, img_size);
		//convolution2D_Y << <DimGrid_Y, DimBlock_Y >> > (d_out_img_x, d_out_img_y, mask_dim, img_size);
		
		// Method 2: Kernel Functions
		convolution2D_X_2 << <DimGrid_X_Tiled, DimBlock_X_Tiled >> > (d_img, d_out_img_x, mask_dim, img_size, dim_X);
		convolution2D_Y_2 << <DimGrid_Y_Tiled, DimBlock_Y_Tiled >> > (d_out_img_x, d_out_img_y, mask_dim, img_size, dim_X, dim_Y);
	}
	printf("GPU Complete\n");

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerConv = msecTotal / iterations;
	float flopsPerConv = 2.0 * 2.0 * img_size * mask_dim;
	float gigaFlops = (flopsPerConv * 1.0e-9f) / (msecPerConv / 1000.0f);
	printf("Performance= %.2f GFlop/s\n Time= %.3f msec\n Size= %.0f Ops\n", gigaFlops, msecPerConv, flopsPerConv);


	checkCudaErrors(cudaMemcpy(h_out_img, d_out_img_y, mem_img_size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_img));
	checkCudaErrors(cudaFree(d_out_img_x));
	checkCudaErrors(cudaFree(d_out_img_y));
	
	if (debug) {
		printf("\nGPU Results:\n");
		for (row = 0; row < img_size; row++) {
			if (row > 0 && row % dim_X == 0) {
				printf("\n");
			}
			printf("%d\t", h_out_img[row]);
		}
	}
	
	if (!no_check) {

		/*
			CPU Convolution
		*/
		printf("\nPerforming CPU Convolution...\n");
		float CPUmsec = 0; 
		clock_t start, end;
		start = clock();
		
		int CPU_iterations = 150;
		for (int j = 0; j < CPU_iterations; j++) {
			convolution_CPU_X(h_img, h_cpu_out_img_x, h_mask, mask_dim, dim_X, dim_Y);
			convolution_CPU_Y(h_cpu_out_img_x, h_cpu_out_img_y, h_mask, mask_dim, dim_X, dim_Y);
		}

		end = clock();
		float cpuTime = (double)(end - start) / CLOCKS_PER_SEC * 1000;

		printf("CPU Complete\n");
		float cpuMegaFlops = (flopsPerConv * 1.0e-6f) / (cpuTime / 1000.0f);
		printf("Performance = %.2f MFlop/s, %.2f msec", cpuMegaFlops, cpuTime);

		if (debug) {
			printf("\nCPU Results:\n");
			for (row = 0; row < img_size; row++) {
				if (row > 0 && row % dim_X == 0) {
					printf("\n");
				}
				printf("%d\t", h_cpu_out_img_y[row]);
			}
			printf("\n");
		}

		// GPU Result Validation
		printf("\nOutput Image Validation...\n");
		bool equlityCheck = checkOutImgs(h_out_img, h_cpu_out_img_y, img_size);
		if (equlityCheck) {
			printf("Equality Check Passed\n");
		}
		else {
			printf("ERROR: Output Images not equal\n");
		}
	}

	free(h_img);
	free(h_out_img);
	free(h_cpu_out_img_x);
	free(h_cpu_out_img_y);
	free(h_mask);
}
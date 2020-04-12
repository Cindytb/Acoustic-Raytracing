
#include "kernels.cuh"
namespace kernels {
	int currDevice = -1;
	cudaDeviceProp prop;
	int numThreads = 256;
	void compute_irs_wrapper(float* d_histogram, float* d_ir, int hbss, size_t size_x, size_t size_y, cudaStream_t stream) {
		int threads = 256;
		int blocks = (hbss * size_x * size_y + threads - 1) / threads;
		compute_irs << < threads, blocks, 0, stream >> > (
			d_histogram,
			d_ir,
			hbss,
			size_x,
			size_y);
	}

	// Must be run once per microphone
	// hbss = histogram bin size sample. Typically 192
	// size_x = size of time axis of histogram. 2.5k by defeault
	// size_y = freq_bands
	__global__ void compute_irs(float* d_histogram, float* d_ir, int hbss, size_t size_x, size_t size_y)
	{
		// x = time axis of histogram. 2.5k by default
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		// y = frequency band of histogram. 8 by default
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

		// Hacky IR implementation that just takes the first band number
		d_ir[x * hbss] = d_histogram[x * size_y];
	}


	__global__ void fillZeros(float* buf, size_t size)
	{
		const size_t numThreads = blockDim.x * gridDim.x;
		const size_t threadID = blockIdx.x * blockDim.x + threadIdx.x;

		for (size_t i = threadID; i < size; i += numThreads)
		{
			buf[i] = 0.0f;
		}
	}
	void fillWithZeroesKernel(float* buf, int size, cudaStream_t s)
	{

		// TODO: Implement this when there are several GPUs/
		// Assuming one GPU for now
		// This code is a scalability measure that
		// ensures that the kernel can fit on the GPU
		// since the histogram can get very large
		/*if (currDevice == -1) {
			checkCudaErrors(cudaGetDevice(&currDevice));
			cudaGetDeviceProperties(&prop, currDevice);
			int maxGridSize[3] = { prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] };

		}*/
		/*
		int device;
		checkCudaErrors(cudaGetDevice(&device));
		if (currDevice == -1 || currDevice != device) {
			currDevice = device;
			cudaGetDeviceProperties(&prop, device);
		}*/

		int numBlocks = (size + numThreads - 1) / numThreads;
		if (numBlocks > 1024) {
			int num_iterations = (numBlocks + 1023) / 1024;
			for (int i = 0; i < num_iterations; i++) {
				int curr_block_size = numBlocks > 1024 ? 1024 : numBlocks;
				int curr_size = curr_block_size == 1024 ? curr_block_size * numThreads : size;
				if (s)
				{
					fillZeros << <numThreads, curr_block_size, 0, s >> > (buf, curr_size);
				}
				else
				{
					fillZeros << <numThreads, curr_block_size >> > (buf, curr_size);
				}
				buf += curr_block_size * numThreads;
				size -= curr_size;
				numBlocks -= curr_block_size;
			}

		}
		else {
			if (s)
			{
				fillZeros << <numThreads, numBlocks, 0, s >> > (buf, size);
			}
			else
			{
				fillZeros << <numThreads, numBlocks >> > (buf, size);
			}
		}

		getLastCudaError("Kernel Launch Failure");
	}
};
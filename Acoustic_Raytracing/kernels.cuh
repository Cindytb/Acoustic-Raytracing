#pragma once
#ifndef __KERNELS__
#define __KERNELS__
/*CUDA Includes*/
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

namespace kernels {
	void fillWithZeroesKernel(float* buf, int size, cudaStream_t s = 0);
	void compute_irs_wrapper(float* d_histogram, float* d_ir, int hbss, size_t size_x, size_t size_y, cudaStream_t stream);
	__global__ void compute_irs(float* d_histogram, float* d_ir, int hbss, size_t size_x, size_t size_y);
}
#endif
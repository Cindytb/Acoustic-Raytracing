#pragma once
#ifndef __KERNELS__
#define __KERNELS__
/*CUDA Includes*/
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

void fillWithZeroesKernel(float *buf, int size, cudaStream_t s = 0);

#endif
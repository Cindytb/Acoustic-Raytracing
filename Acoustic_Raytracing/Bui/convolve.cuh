#pragma once
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <locale.h>
#include <stdbool.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// thrust includes
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <cmath>

namespace cufft {
    typedef float2 Complex;
    void convolve(float* d_ibuf, float* d_rbuf, long long paddedSize);
    void normalize_fft(float* d_input, size_t padded_size);
    void declip(float* d_input, size_t padded_size);
    void forward_fft_wrapper(float* d_input, size_t padded_size);
	void inverse_fft_wrapper(float* d_input, size_t padded_size);
    void convolve_ifft_wrapper(cufftComplex* d_ibuf, cufftComplex* d_rbuf, size_t padded_size);
// cuFFT API errors
#ifndef CHECK_CUFFT_ERRORS
#define CHECK_CUFFT_ERRORS(call) { \
    cufftResult_t err; \
    if ((err = (call)) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _cudaGetErrorEnum(err), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}
#endif
}
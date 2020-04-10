
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "optix7.h"

/*
    This is a short header file and inline function 
    to fully synchronize all GPU processes and check for errors.
    This allows for scalability so the programmer can put
    DEBUG_CHECK() in any applicable places during debugging 
    and then set the FULL_CUDA_DEBUG flag to false when ready
    for release and realtime functionality.
*/
#define FULL_CUDA_DEBUG 1

inline void DEBUG_CHECK()
{
    if (FULL_CUDA_DEBUG)
    {
        checkCudaErrors(cudaDeviceSynchronize());
        CUDA_SYNC_CHECK();
    }
}
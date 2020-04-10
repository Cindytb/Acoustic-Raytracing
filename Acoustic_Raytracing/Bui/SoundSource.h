#pragma once

#include "SoundItem.h"
#include "Microphone.h"
#include "../LaunchParams.h"
#include "../kernels.cuh"
#include "convolve.cuh"
#include "../debug.cuh"
#include "constants.h"
#include "optix.h"
#include "gdt/math/vec.h"
//#include "optix7.h"

/* Library Includes */
#include <iostream>
#include <fstream>
//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>
//#include <math.h>
//#include <algorithm>
//#include <ctime>
#include <chrono>

/*CUDA Includes*/
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

/* CUFFT */
#include <cufft.h>
#include <cufftXt.h>
/* libsdnfile*/
#include <sndfile.hh>
class SoundSource : SoundItem
{
public:
	SoundSource();
	SoundSource(gdt::vec3f pos);
	SoundSource(gdt::vec3f pos, gdt::vec3f orientation);
	~SoundSource();
	
	void add_mic(Microphone mic);
	void trace();
	void compute_IRs();
	void convolve();
	void convolve_file(std::string input_file,
		std::string output_file,
		int mic_no);
	osc::LaunchParams* local_histogram;
private:

	osc::LaunchParams* d_local_histogram;
	float* m_histogram;
	float* m_irs;
	int m_ir_nonzero_length;
	std::vector<Microphone> m_microphones;
	cudaStream_t m_stream;
	SndfileHandle* file;
};
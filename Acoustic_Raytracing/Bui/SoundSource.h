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
	
	void add_mic(Microphone &mic);
	void trace();
	void compute_IRs();
	void convolve();
	void convolve_file(std::string input_file,
		std::string output_file,
		int mic_no);
	void add_buffer(float *input);
	void HACK_upload_ir(std::string input_file);
private:
	/* Launch data, host & device*/
	osc::LaunchData* m_local_histogram;
	osc::LaunchData* d_local_histogram;
	
	/* Rolling input for realtime convolution*/
	float* m_buffered_input;
	float* d_buffered_input;
	size_t m_buffer_size;

	/* Individual microphone data*/
	std::vector<Microphone> m_microphones;
	std::vector<float*> m_histogram;
	std::vector<float*> m_irs;
	std::vector<float*> d_irs;
	std::vector<float*> d_conv_bufs;
	std::vector<float*> m_summing_bus;

	cudaStream_t m_stream;
	bool scene_change;

	/* Future: moving away from a histogram*/
	float* d_transfer_function;
};
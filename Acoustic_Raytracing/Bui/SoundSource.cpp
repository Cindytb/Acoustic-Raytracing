#pragma once

#include "SoundSource.h"
int next_pow_2(int v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

SoundSource::SoundSource()
{
	new (this) SoundSource({ 0, 0, 0 }, { 0, 0, 0 });
}
SoundSource::SoundSource(gdt::vec3f pos)
{
	new (this) SoundSource(pos, { 0, 0, 0 });
}

SoundSource::SoundSource(gdt::vec3f pos, gdt::vec3f orientation)
{
	m_position = pos;
	m_orientation = orientation;
	m_local_histogram = new osc::LaunchData();
	num_src++;
	int hist_bin_size_samples = fs * hist_bin_size;
	hist_bin_size = hist_bin_size_samples / (float)fs;
	dist_thres = c * time_thres;
	freq_bands = (int)log2(fs / 125.0);
	time_bins = fs * time_thres * hist_bin_size;
	checkCudaErrors(cudaStreamCreate(&m_stream));

	/*An absolutely gigantic buffer of how much audio to store*/
	m_buffer_size = next_pow_2(time_thres * fs + FRAMES_PER_BUFFER - 1);
	m_buffered_input = new float[m_buffer_size];
#pragma omp parallel for
	for (size_t i = 0; i < m_buffer_size; i++)
	{
		m_buffered_input[i] = 0.0f;
	}
	checkCudaErrors(cudaMalloc(&d_buffered_input, (m_buffer_size + 2) * sizeof(float)));
	scene_change = true;

	checkCudaErrors(cudaMalloc(&d_transfer_function, (m_buffer_size + 2) * sizeof(float)));

	kernels::fillWithZeroesKernel(d_transfer_function, m_buffer_size + 2);
	kernels::fillWithZeroesKernel(d_buffered_input, m_buffer_size + 2);

	checkCudaErrors(
		cudaMalloc(&m_local_histogram->d_histogram,
			MAX_NUM_MICS * time_bins * freq_bands * sizeof(float)));
	checkCudaErrors(cudaMalloc(&m_local_histogram->d_transmitted,
		freq_bands * num_rays * sizeof(float)));
	DEBUG_CHECK();
	checkCudaErrors(
		cudaMalloc((void**)&d_local_histogram, sizeof(osc::LaunchData)));
	kernels::fillWithZeroesKernel(m_local_histogram->d_histogram,
		MAX_NUM_MICS * time_bins * freq_bands,
		m_stream);
	kernels::fillWithZeroesKernel(
		m_local_histogram->d_transmitted, freq_bands * num_rays, m_stream);
}

void SoundSource::add_mic(Microphone* mic)
{
	m_microphones.push_back(mic);
	m_histogram.push_back(new float[time_bins * freq_bands]);
	m_irs.push_back(new float[(size_t)(time_thres * fs)]);
	m_summing_bus.push_back(new float[frames_per_buffer]);
	float* d_ir, *d_conv;
	checkCudaErrors(cudaMalloc(&d_ir, (m_buffer_size + 2) * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_conv, (m_buffer_size + 2) * sizeof(float)));
	d_irs.push_back(d_ir);
	d_conv_bufs.push_back(d_conv);
#pragma omp parallel for
	for (int i = 0; i < (size_t)(time_thres * fs); i++)
	{
		m_irs[m_microphones.size() - 1][i] = 0.0f;
	}
	kernels::fillWithZeroesKernel(d_ir, m_buffer_size + 2, m_stream);
	kernels::fillWithZeroesKernel(d_conv, m_buffer_size + 2, m_stream);
}

void SoundSource::trace()
{
	m_local_histogram->freq_bands = freq_bands;
	m_local_histogram->pos = m_position;
	m_local_histogram->orientation = m_orientation;
	m_local_histogram->traversable = traversable;
	m_local_histogram->time_bins = time_bins;
	m_local_histogram->hist_bin_size = hist_bin_size;
	m_local_histogram->dist_thres = dist_thres;
	m_local_histogram->energy_thres = energy_thres;
	m_local_histogram->c = c;
	m_local_histogram->fs = fs;
	m_local_histogram->d_transfer_function = d_transfer_function;
	m_local_histogram->buffer_size = m_buffer_size;
	printf("num_rays: %i\n", num_rays);
	checkCudaErrors(cudaMemcpy(d_local_histogram,
		m_local_histogram,
		sizeof(osc::LaunchData),
		cudaMemcpyHostToDevice));

	auto start = std::chrono::high_resolution_clock::now();
	OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
		pipeline,
		m_stream,
		/*! parameters and SBT */
		(CUdeviceptr)d_local_histogram,
		sizeof(osc::LaunchData),
		&sbt,
		/*! dimensions of the launch: */
		num_rays,
		1,
		1));
	DEBUG_CHECK();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "Time taken by GPU function: "
		<< duration.count() << " microseconds" << std::endl;
	for (int i = 0; i < num_mics; i++)
	{
		checkCudaErrors(cudaMemcpyAsync(m_histogram[i],
										m_local_histogram->d_histogram + i * time_bins * freq_bands,
										time_bins * freq_bands * sizeof(float),
										cudaMemcpyDeviceToHost,
										m_stream));
	}

	DEBUG_CHECK();

	// Used to do precision benchmarking between this implmentation
	// and pyroomacoustics
	// std::ofstream myfile;
	// myfile.open("histogram.dump");

	// for (int i = 0; i < time_bins; i++)
	// {
	// 	myfile << m_histogram[0][i * freq_bands] << std::endl;
	// }
	// myfile.close();
}

void SoundSource::compute_IRs()
{
	std::cout<< "computing IRS" << std::endl;
	int max_ir_length = (int)time_thres * fs;
	int hist_bin_size_samples = SoundItem::fs * hist_bin_size;
	for (int i = 0; i < num_mics; i++)
	{
		
		kernels::compute_irs_wrapper(m_local_histogram->d_histogram + i * time_bins * freq_bands,
			d_irs[i],
			hist_bin_size_samples,
			time_bins,
			freq_bands,
			m_stream);
	}
	checkCudaErrors(cudaStreamSynchronize(m_stream));

#pragma omp parallel for
	for (int mic_no = 0; mic_no < num_mics; mic_no++)
	{
#pragma omp parallel for
		for (int i = 0; i < time_bins; i++)
		{
			m_irs[mic_no][i * hist_bin_size_samples] = m_histogram[mic_no][i * freq_bands];
		}
	}

	/*std::ofstream myfile;
	myfile.open("ir.dump");

	for (int i = 0; i < max_ir_length; i++) {
		myfile << m_irs[i] << std::endl;
	}
	myfile.close();*/
}


void SoundSource::add_buffer(float* input)
{
	/*Copy into current buffer into x */
	memcpy(
		m_buffered_input + (m_buffer_size - frames_per_buffer), /*Go to the end and work backwards*/
		input,
		frames_per_buffer * sizeof(float));
	checkCudaErrors(cudaMemcpy(d_buffered_input, m_buffered_input, m_buffer_size * sizeof(float), cudaMemcpyHostToDevice));
	cufft::forward_fft_wrapper(d_buffered_input, m_buffer_size);
	if (scene_change)
	{
		trace();
		//compute_IRs();
		//for (int i = 0; i < num_mics; i++) {
			//cufft::forward_fft_wrapper(d_irs[i], m_buffer_size);
		//}
		// TODO: Currently only supports 1 microphone input
		checkCudaErrors(cudaMemcpy(d_irs[0], d_transfer_function, (m_buffer_size + 2) * sizeof(float), cudaMemcpyDeviceToDevice));
		scene_change = false;
	}
	for(int i = 0; i < num_mics; i++){
		cufft::convolve_ifft_wrapper((cufftComplex*)d_buffered_input, (cufftComplex*)d_irs[i], (cufftComplex*)d_conv_bufs[i], m_buffer_size);
		cufft::normalize_fft(d_conv_bufs[i], m_buffer_size);
		checkCudaErrors(cudaMemcpy(m_summing_bus[i], 
			d_conv_bufs[i] + m_buffer_size - frames_per_buffer, 
			frames_per_buffer * sizeof(float), 
			cudaMemcpyDeviceToHost
			));
		checkCudaErrors(cudaDeviceSynchronize());
		float* output = m_microphones[i]->get_output();
		// TODO: Turn this into an atomic addition to prevent data races
		for(int j = 0; j < frames_per_buffer; j++){
			output[j] += m_summing_bus[i][j];
		}
	}
	/*Overlap-save*/
	memmove(
		m_buffered_input,
		m_buffered_input + frames_per_buffer,
		sizeof(float) * (m_buffer_size - frames_per_buffer));
}








void TDconvolution(float* ibuf, float* rbuf, size_t iframes, size_t rframes, float* obuf)
{
	int oframes = iframes + rframes - 1;
#pragma omp parallel for
	for (int i = 0; i < oframes; i++)
	{
		obuf[i] = 0.0f;
	}
	for (size_t k = 0; k < rframes; k++)
	{
		for (size_t n = 0; n < iframes; n++)
		{
			obuf[k + n] += ibuf[n] * rbuf[k];
		}
	}
}
void SoundSource::export_impulse_response(std::string filename,
	int mic_no)
{
	int ir_length = (int)fs * time_thres;
	float* r_buf = new float[m_buffer_size];
	checkCudaErrors(cudaMemcpy(r_buf, d_transfer_function, m_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
	cufft::normalize_fft(d_transfer_function, m_buffer_size);
	checkCudaErrors(cudaMemcpy(r_buf, d_transfer_function, m_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
	cufft::inverse_fft_wrapper(d_transfer_function, m_buffer_size);
	checkCudaErrors(cudaMemcpy(r_buf, d_transfer_function, m_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
	cufft::declip(d_transfer_function, m_buffer_size);
	checkCudaErrors(cudaMemcpy(r_buf, d_transfer_function, m_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = ir_length - 1; i >= 0; i--) {
		if (fabs(r_buf[i]) > 1e-8) {
			ir_length = i + 1;
			break;
		}
	}
	SndfileHandle ofile = SndfileHandle(filename, SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_24, 1, fs);

	size_t count = ofile.write(r_buf, ir_length);
	delete[] r_buf;
}

void SoundSource::convolve_file(std::string input_file,
	std::string output_file,
	int mic_no)
{
	SndfileHandle ifile;
	ifile = SndfileHandle(input_file);
	if (ifile.channels() != 1)
	{
		printf("ERROR: only mono files allowed in this function\n");
		exit(EXIT_FAILURE);
	}
	if (ifile.samplerate() != fs)
	{
		printf("ERROR: Wrong sample rate\n");
		exit(EXIT_FAILURE);
	}
	int ir_length = (int)fs * time_thres;
	size_t oframes = ifile.frames() + ir_length - 1;
	size_t padded_size = next_pow_2(oframes);
	float* input_buf, * d_input, * d_filter;
	input_buf = new float[padded_size];
	float* r_buf = new float[m_buffer_size];
#pragma omp parallel for
	for (int i = 0; i < padded_size; i++)
	{
		input_buf[i] = 0.0f;
	}
	ifile.readf(input_buf, ifile.frames());
	checkCudaErrors(cudaMalloc(&d_input, (padded_size + 2) * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_filter, (padded_size + 2) * sizeof(float)));
	kernels::fillWithZeroesKernel(d_input, padded_size + 2);
	kernels::fillWithZeroesKernel(d_filter, padded_size + 2);
	//cufft::normalize_fft(d_transfer_function, m_buffer_size);
	checkCudaErrors(cudaMemcpy(r_buf, d_transfer_function, m_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
	cufft::inverse_fft_wrapper(d_transfer_function, m_buffer_size);
	checkCudaErrors(cudaMemcpy(r_buf, d_transfer_function, m_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
	cufft::declip(d_transfer_function, m_buffer_size);
	checkCudaErrors(cudaMemcpy(r_buf, d_transfer_function, m_buffer_size * sizeof(float), cudaMemcpyDeviceToHost)); 
	
	checkCudaErrors(cudaMemcpy(d_input, input_buf, ifile.frames() * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filter, d_transfer_function, ir_length * sizeof(float), cudaMemcpyDeviceToDevice));

	cufft::convolve(d_input, d_filter, padded_size);
	checkCudaErrors(cudaDeviceSynchronize());
	cufft::normalize_fft(d_input, padded_size);
	checkCudaErrors(cudaDeviceSynchronize());
	//cufft::declip(d_input, padded_size);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(input_buf, d_input, oframes * sizeof(float), cudaMemcpyDeviceToHost));
	float max = 0;

	for (int i = 0; i < oframes; i++)
	{
		float local_val = fabs(input_buf[i]);
		if (local_val > max)
		{
			max = local_val;
		}
	}
#pragma omp parallel for
	for (int i = 0; i < oframes; i++)
	{
		input_buf[i] /= max;
	}
	SndfileHandle ofile = SndfileHandle(output_file, SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_24, 1, fs);

	size_t count = ofile.write(input_buf, oframes);
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_filter));
	delete[] input_buf;
}
SoundSource::~SoundSource()
{
	num_src--;
	delete[] m_buffered_input;
	checkCudaErrors(cudaFree(m_local_histogram->d_histogram));
	for (int i = 0; i < num_mics; i++)
	{
		delete[] m_histogram[i];
		delete[] m_irs[i];
		checkCudaErrors(cudaFree(d_conv_bufs[i]));
		checkCudaErrors(cudaFree(d_irs[i]));
	}
	checkCudaErrors(cudaFree(m_local_histogram->d_transmitted));
	checkCudaErrors(cudaFree(d_local_histogram));
}
void SoundSource::HACK_upload_ir(std::string input_file) {
	SndfileHandle ifile;
	ifile = SndfileHandle(input_file);
	if (ifile.channels() != 1)
	{
		printf("ERROR: only mono files allowed in this function\n");
		exit(EXIT_FAILURE);
	}
	if (ifile.samplerate() != fs)
	{
		printf("ERROR: Wrong sample rate\n");
		exit(EXIT_FAILURE);
	}

	ifile.readf(m_irs[0], ifile.frames());
	checkCudaErrors(cudaMemcpy(d_irs[0], m_irs[0], ifile.frames() * sizeof(float), cudaMemcpyHostToDevice));
	cufft::forward_fft_wrapper(d_irs[0], m_buffer_size);
}
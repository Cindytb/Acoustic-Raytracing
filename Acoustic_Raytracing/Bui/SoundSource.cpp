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
	local_histogram = new osc::LaunchParams();
	num_src++;
	int hist_bin_size_samples = fs * hist_bin_size;
	hist_bin_size = hist_bin_size_samples / (float)fs;
	dist_thres = c * time_thres;
	freq_bands = (int)log2(fs / 125.0);
	time_bins = fs * time_thres * hist_bin_size;
	checkCudaErrors(cudaStreamCreate(&m_stream));

	checkCudaErrors(
		cudaMalloc(&local_histogram->d_histogram,
			MAX_NUM_MICS * time_bins * freq_bands * sizeof(float)));
	checkCudaErrors(cudaMalloc(&local_histogram->d_transmitted,
		freq_bands * num_rays * sizeof(float)));
	m_histogram = new float[MAX_NUM_MICS * time_bins * freq_bands];
	size_t size = MAX_NUM_MICS * time_thres * fs;
	m_irs = new float[size]; // maximum length of an IR
	for (int i = 0; i < MAX_NUM_MICS * time_thres * fs; i++)
	{
		m_irs[i] = 0.0f;
	}
	DEBUG_CHECK();
	checkCudaErrors(
		cudaMalloc((void**)&d_local_histogram, sizeof(osc::LaunchParams)));
	fillWithZeroesKernel(local_histogram->d_histogram,
		MAX_NUM_MICS * time_bins * freq_bands,
		m_stream);
	fillWithZeroesKernel(
		local_histogram->d_transmitted, freq_bands * num_rays, m_stream);
}

void SoundSource::add_mic(Microphone mic)
{
	m_microphones.push_back(mic);
}

void SoundSource::trace()
{
	local_histogram->freq_bands = freq_bands;
	local_histogram->pos = m_position;
	local_histogram->orientation = m_orientation;
	local_histogram->traversable = traversable;
	local_histogram->time_bins = time_bins;
	local_histogram->hist_bin_size = hist_bin_size;
	local_histogram->dist_thres = dist_thres;
	local_histogram->energy_thres = energy_thres;
	local_histogram->c = c;
	printf("num_rays: %i\n", num_rays);
	checkCudaErrors(cudaMemcpy(d_local_histogram,
		local_histogram,
		sizeof(osc::LaunchParams),
		cudaMemcpyHostToDevice));


	auto start = std::chrono::high_resolution_clock::now();
	OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
		pipeline,
		m_stream,
		/*! parameters and SBT */
		(CUdeviceptr)d_local_histogram,
		sizeof(osc::LaunchParams),
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

	checkCudaErrors(cudaMemcpyAsync(m_histogram,
		local_histogram->d_histogram,
		MAX_NUM_MICS * time_bins * freq_bands * sizeof(float),
		cudaMemcpyDeviceToHost,
		m_stream));
	DEBUG_CHECK();

	// Used to do precision benchmarking between this implmentation
	// and pyroomacoustics
	std::ofstream myfile;
	myfile.open("histogram.dump");

	for (int i = 0; i < time_bins; i++) {
		myfile << m_histogram[i * freq_bands] << std::endl;
	}
	myfile.close();
}

void SoundSource::compute_IRs()
{
	int hist_bin_size_samples = SoundItem::fs * hist_bin_size;
	checkCudaErrors(
		cudaMemcpyAsync(m_histogram,
			local_histogram->d_histogram,
			MAX_NUM_MICS * time_bins * freq_bands * sizeof(float),
			cudaMemcpyDeviceToHost,
			m_stream));
	DEBUG_CHECK();
	checkCudaErrors(cudaStreamSynchronize(m_stream));

	int max_ir_length = (int)time_thres * fs;
	for (int mic_no = 0; mic_no < num_mics; mic_no++)
	{
		for (int i = 0; i < time_bins; i++)
		{
			m_irs[mic_no * max_ir_length + i * hist_bin_size_samples] = m_histogram[mic_no * freq_bands * time_bins + i * freq_bands];
		}
	}

	/*std::ofstream myfile;
	myfile.open("ir.dump");

	for (int i = 0; i < max_ir_length; i++) {
		myfile << m_irs[i] << std::endl;
	}
	myfile.close();*/
}

void TDconvolution(float* ibuf, float* rbuf, size_t iframes, size_t rframes, float* obuf) {
	int oframes = iframes + rframes - 1;

	for (int i = 0; i < oframes; i++) {
		obuf[i] = 0.0f;
	}
	for (size_t k = 0; k < rframes; k++) {
		for (size_t n = 0; n < iframes; n++) {
			obuf[k + n] += ibuf[n] * rbuf[k];
		}
	}
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
	if (ifile.samplerate() != fs) {
		printf("ERROR: Wrong sample rate\n");
		exit(EXIT_FAILURE);
	}
	int ir_length = 0;
	int max_ir_length = (int)time_thres * fs;
	for (int i = 0; i < max_ir_length; i++)
	{
		if (m_irs[mic_no * max_ir_length + i] > 0 || m_irs[mic_no * max_ir_length + i] < 0)
		{
			ir_length = i;
		}
	}
	// Ratchet integrator
	int hist_bin_size_samples = SoundItem::fs * hist_bin_size;
	float* ir = new float[ir_length + hist_bin_size_samples - 1];
	float* ratchet_integrator = new float[hist_bin_size_samples];
	for (int i = 0; i < hist_bin_size_samples; i++) {
		ratchet_integrator[i] = 1.0 / hist_bin_size_samples;
	}
	TDconvolution(m_irs + mic_no * max_ir_length, ratchet_integrator, ir_length, hist_bin_size_samples, ir);
	float max_val = 0;
	for (int i = 0; i < ir_length; i++) {
		float local_val = fabs(ir[i]);
		if (local_val > max_val) {
			max_val = local_val;
		}
	}
#pragma omp parallel for
	for (int i = 0; i < ir_length; i++) {
		ir[i] /= max_val;
	}
	size_t oframes = ifile.frames() + ir_length - 1;
	size_t padded_size = next_pow_2(oframes);
	float* input_buf, * d_input, * d_filter;
	input_buf = new float[padded_size];
#pragma omp parallel for
	for (int i = 0; i < padded_size; i++) {
		input_buf[i] = 0.0f;
	}
	ifile.readf(input_buf, ifile.frames());
	checkCudaErrors(cudaMalloc(&d_input, (padded_size + 2) * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_filter, (padded_size + 2) * sizeof(float)));
	fillWithZeroesKernel(d_input, padded_size + 2);
	fillWithZeroesKernel(d_filter, padded_size + 2);
	checkCudaErrors(cudaMemcpy(d_input, input_buf, ifile.frames() * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filter, m_irs + mic_no * max_ir_length, ir_length * sizeof(float), cudaMemcpyHostToDevice));

	cufft::convolve(d_input, d_filter, padded_size);
	checkCudaErrors(cudaDeviceSynchronize());
	cufft::normalize_fft(d_input, padded_size);
	checkCudaErrors(cudaDeviceSynchronize());
	//cufft::declip(d_input, padded_size);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(input_buf, d_input, oframes * sizeof(float), cudaMemcpyDeviceToHost));
	float max = 0;

	for (int i = 0; i < oframes; i++) {
		float local_val = fabs(input_buf[i]);
		if (local_val > max) {
			max = local_val;
		}
	}
#pragma omp parallel for
	for (int i = 0; i < oframes; i++) {
		input_buf[i] /= max;
	}
	SndfileHandle ofile = SndfileHandle(output_file, SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_24, 1, fs);

	size_t count = ofile.write(input_buf, oframes);
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_filter));
	delete[] input_buf;
}
void SoundSource::convolve()
{
}
SoundSource::~SoundSource()
{
	num_src--;
	checkCudaErrors(cudaFree(local_histogram->d_histogram));
	delete[] m_histogram;
	checkCudaErrors(cudaFree(local_histogram->d_transmitted));
	checkCudaErrors(cudaFree(d_local_histogram));
}